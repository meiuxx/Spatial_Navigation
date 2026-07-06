// AgentCommandUI.cs
//
// Self-contained UI panel — builds its own Canvas, Panel, InputField, buttons,
// and status label entirely at runtime using Unity's built-in UI (no TMP).
// Just attach this script to ANY GameObject in the scene and press Play.
//
// REQUIRES: nothing beyond Unity's built-in UI package (always present).
//
// NETWORK: two separate Python endpoints.
//   - llm.py   on 127.0.0.1:5012 (UNITY_LLM_PORT) — natural-language "Go"
//              queries. llm.py does CLIP ranking + LLM grounding against the
//              saved landmark graph, then drives the agent itself by calling
//              main.py internally. Reply is a pre-formatted status string.
//   - main.py  on 127.0.0.1:5010 (LLM_CMD_PORT)   — "Stop" only. main.py's
//              command server no longer accepts raw scene/landmark queries;
//              all query grounding happens in llm.py.
//
// ROUTING (single input field):
//   - Go button   → raw query text sent to llm.py, e.g. "lobby" or "IV pole"
//   - Stop button  → {"command":"stop"} sent to main.py

using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

public class AgentCommandUI : MonoBehaviour
{
    // ── Network config (edit here or expose as public fields) ─────────────────
    // main.py — only handles "stop" now (query grounding lives in llm.py).
    public string pythonHost = "127.0.0.1";
    public int pythonPort = 5010;

    // llm.py — natural-language navigation queries (CLIP + LLM grounding).
    public string llmHost = "127.0.0.1";
    public int llmPort = 5012;

    // ── Runtime state ─────────────────────────────────────────────────────────
    private InputField _inputField;
    private Button _goButton;
    private Button _stopButton;
    private Text _statusText;
    private bool _busy = false;
    private string _pendingReply;
    private bool _replyReady = false;
    private ReplyKind _pendingKind;

    private enum ReplyKind { Stop, Query }

    // Status colours
    private static readonly Color ColIdle = new Color(0.75f, 0.75f, 0.75f);
    private static readonly Color ColSending = new Color(0.95f, 0.85f, 0.20f);
    private static readonly Color ColArrived = new Color(0.25f, 0.85f, 0.40f);
    private static readonly Color ColAbandoned = new Color(0.95f, 0.55f, 0.20f);
    private static readonly Color ColError = new Color(0.95f, 0.30f, 0.30f);
    private static readonly Color ColStopped = new Color(0.65f, 0.65f, 0.95f);

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    private void Awake()
    {
        _BuildUI();
    }

    private void Update()
    {
        if (!_replyReady) return;
        _replyReady = false;
        if (_pendingKind == ReplyKind.Query) _HandleQueryReply(_pendingReply);
        else _HandleStopReply(_pendingReply);
        _pendingReply = null;
        _SetBusy(false);
    }

    // ── UI builder ────────────────────────────────────────────────────────────

    // Layer 5 is Unity's built-in "UI" layer. We strip it from every camera's
    // culling mask so the panel never appears in any RenderTexture the RGB
    // camera or OCR pipeline reads from.
    private const int UILayer = 5;

    private static void _SetLayerRecursive(GameObject go, int layer)
    {
        go.layer = layer;
        foreach (Transform child in go.transform)
            _SetLayerRecursive(child.gameObject, layer);
    }

    private void _BuildUI()
    {
        // Strip UI layer from every camera BEFORE creating the canvas so there
        // is never a frame where the UI bleeds into a camera RenderTexture.
        foreach (var cam in FindObjectsOfType<Camera>())
            cam.cullingMask &= ~(1 << UILayer);

        // ── Canvas ────────────────────────────────────────────────────────────
        var canvasGO = new GameObject("AgentCommandCanvas");
        canvasGO.layer = UILayer;

        var canvas = canvasGO.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 100;
        canvasGO.AddComponent<CanvasScaler>().uiScaleMode =
            CanvasScaler.ScaleMode.ScaleWithScreenSize;
        canvasGO.AddComponent<GraphicRaycaster>();

        // EventSystem (needed for clicks and keyboard)
        if (FindObjectOfType<UnityEngine.EventSystems.EventSystem>() == null)
        {
            var es = new GameObject("EventSystem");
            es.AddComponent<UnityEngine.EventSystems.EventSystem>();
            es.AddComponent<UnityEngine.EventSystems.StandaloneInputModule>();
        }

        // ── Panel (semi-transparent background) ───────────────────────────────
        var panel = _MakeRect("Panel", canvasGO.transform);
        var panelImg = panel.AddComponent<Image>();
        panelImg.color = new Color(0.08f, 0.08f, 0.12f, 0.92f);

        // Anchor to top-right, fixed size 420 × 166
        var panelRT = panel.GetComponent<RectTransform>();
        panelRT.anchorMin = panelRT.anchorMax = panelRT.pivot = new Vector2(1f, 1f);
        panelRT.anchoredPosition = new Vector2(-12f, -12f);
        panelRT.sizeDelta = new Vector2(420f, 166f);

        // ── Title ─────────────────────────────────────────────────────────────
        var title = _MakeText("Title", panel.transform, "Agent Navigator",
                              18, FontStyle.Bold, TextAnchor.MiddleLeft);
        _SetRect(title, new Vector2(0, 1), new Vector2(1, 1),
                 new Vector2(14, -36), new Vector2(-14, -8));

        // ── Input field ───────────────────────────────────────────────────────
        var inputGO = _MakeRect("QueryInput", panel.transform);
        var inputImg = inputGO.AddComponent<Image>();
        inputImg.color = new Color(0.18f, 0.18f, 0.24f, 1f);
        _SetRect(inputGO, new Vector2(0, 1), new Vector2(1, 1),
                 new Vector2(14, -78), new Vector2(-130, -42));

        _inputField = inputGO.AddComponent<InputField>();
        _inputField.transition = Selectable.Transition.ColorTint;

        // Placeholder
        var ph = _MakeText("Placeholder", inputGO.transform,
                           "lobby   /   a hospital bed   /   IV pole…",
                           13, FontStyle.Italic, TextAnchor.MiddleLeft);
        ph.GetComponent<Text>().color = new Color(0.45f, 0.45f, 0.55f);
        _FillRect(ph);
        _PadRect(ph, 8, 0, 4, 0);

        // Text component
        var inputText = _MakeText("Text", inputGO.transform,
                                  "", 13, FontStyle.Normal, TextAnchor.MiddleLeft);
        inputText.GetComponent<Text>().color = Color.white;
        _FillRect(inputText);
        _PadRect(inputText, 8, 0, 4, 0);

        _inputField.placeholder = ph.GetComponent<Text>();
        _inputField.textComponent = inputText.GetComponent<Text>();
        _inputField.onEndEdit.AddListener(val => { if (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter)) OnGo(); });

        // ── Go button ─────────────────────────────────────────────────────────
        _goButton = _MakeButton("GoButton", panel.transform, "Go",
                                new Color(0.18f, 0.50f, 0.90f));
        _SetRect(_goButton.gameObject,
                 new Vector2(1, 1), new Vector2(1, 1),
                 new Vector2(-118, -78), new Vector2(-14, -42));
        _goButton.onClick.AddListener(OnGo);

        // ── Stop button ───────────────────────────────────────────────────────
        _stopButton = _MakeButton("StopButton", panel.transform, "■  Stop",
                                  new Color(0.70f, 0.18f, 0.18f));
        _SetRect(_stopButton.gameObject,
                 new Vector2(0, 1), new Vector2(1, 1),
                 new Vector2(14, -122), new Vector2(-14, -86));
        _stopButton.onClick.AddListener(OnStop);

        // ── Status label ──────────────────────────────────────────────────────
        var statusGO = _MakeText("Status", panel.transform,
                                 "Ready", 12, FontStyle.Normal, TextAnchor.MiddleLeft);
        _statusText = statusGO.GetComponent<Text>();
        _statusText.color = ColIdle;
        _SetRect(statusGO, new Vector2(0, 0), new Vector2(1, 0),
                 new Vector2(14, 8), new Vector2(-14, 30));
    }

    // ── Button callbacks ──────────────────────────────────────────────────────

    private void OnGo()
    {
        if (_busy) return;
        string query = _inputField != null ? _inputField.text.Trim() : "";
        if (string.IsNullOrEmpty(query))
        {
            _SetStatus("Enter a room name or object description.", ColError);
            return;
        }

        _SendQuery(query);
    }

    private void OnStop()
    {
        _busy = false;
        _SendCommand("{\"command\":\"stop\"}", "Stopping…");
    }

    // ── Networking ────────────────────────────────────────────────────────────

    // "Stop" — JSON command to main.py, JSON reply.
    private void _SendCommand(string json, string statusMsg) =>
        _Send(json + "\n", pythonHost, pythonPort, statusMsg, ReplyKind.Stop);

    // Natural-language "Go" — raw text query to llm.py, plain-text reply.
    private void _SendQuery(string query) =>
        _Send(query + "\n", llmHost, llmPort, $"Searching for: \"{query}\"", ReplyKind.Query);

    private void _Send(string payload, string host, int port, string statusMsg, ReplyKind kind)
    {
        _SetBusy(true);
        _SetStatus(statusMsg, ColSending);
        _pendingKind = kind;

        var t = new Thread(() =>
        {
            string reply;
            try
            {
                using var client = new TcpClient();
                client.Connect(host, port);
                client.SendTimeout = 3000;
                client.ReceiveTimeout = 120000;

                var stream = client.GetStream();
                var bytes = Encoding.UTF8.GetBytes(payload);
                stream.Write(bytes, 0, bytes.Length);

                var sb = new StringBuilder();
                var buf = new byte[4096];
                while (true)
                {
                    int n = stream.Read(buf, 0, buf.Length);
                    if (n <= 0) break;
                    sb.Append(Encoding.UTF8.GetString(buf, 0, n));
                    if (sb.ToString().Contains("\n")) break;
                }
                reply = sb.ToString().Trim();
            }
            catch (Exception ex)
            {
                reply = kind == ReplyKind.Query
                    ? $"✗ Error: {ex.Message}"
                    : $"{{\"status\":\"error\",\"reason\":\"{_Esc(ex.Message)}\"}}";
                Debug.LogWarning($"[AgentCommandUI] {ex.Message}");
            }
            _pendingReply = reply;
            _replyReady = true;
        });
        t.IsBackground = true;
        t.Start();
    }

    // ── Reply handling ────────────────────────────────────────────────────────

    // llm.py replies with a pre-formatted plain-text status string (see
    // format_result() in llm.py) — no JSON to parse, just pick a colour
    // from the leading glyph.
    private void _HandleQueryReply(string raw)
    {
        if (string.IsNullOrEmpty(raw)) { _SetStatus("No response.", ColError); return; }

        Color col = ColIdle;
        if (raw.StartsWith("✓")) col = ColArrived;
        else if (raw.StartsWith("⚠")) col = ColAbandoned;
        else if (raw.StartsWith("✗")) col = ColError;
        else if (raw.StartsWith("↩") || raw.StartsWith("■")) col = ColStopped;

        _SetStatus(raw.Length > 100 ? raw[..100] + "…" : raw, col);
    }

    // main.py's stop handler only ever replies {"status":"stopped"}; the
    // error case covers connection failures from the catch block above.
    private void _HandleStopReply(string raw)
    {
        if (string.IsNullOrEmpty(raw)) { _SetStatus("No response.", ColError); return; }

        string status = _JsonStr(raw, "status");
        string reason = _JsonStr(raw, "reason");

        if (status == "stopped") _SetStatus("Stopped.", ColStopped);
        else if (status == "error") _SetStatus($"Error: {reason ?? raw}", ColError);
        else _SetStatus(raw.Length > 100 ? raw[..100] + "…" : raw, ColIdle);
    }

    // ── UI state helpers ──────────────────────────────────────────────────────

    private void _SetStatus(string msg, Color col)
    {
        if (_statusText == null) return;
        _statusText.text = msg;
        _statusText.color = col;
    }

    private void _SetBusy(bool busy)
    {
        _busy = busy;
        if (_goButton != null) _goButton.interactable = !busy;
        if (_stopButton != null) _stopButton.interactable = true;
    }

    // ── Minimal JSON read/write helpers ───────────────────────────────────────

    private static string _JsonStr(string json, string key)
    {
        string search = $"\"{key}\":\"";
        int s = json.IndexOf(search, StringComparison.Ordinal);
        if (s < 0) return null;
        s += search.Length;
        int e = json.IndexOf('"', s);
        return e < 0 ? null : json.Substring(s, e - s);
    }

    private static string _Esc(string s) =>
        s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", " ");

    // ── Unity UI factory helpers ──────────────────────────────────────────────

    private static GameObject _MakeRect(string name, Transform parent)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.layer = UILayer;
        go.transform.SetParent(parent, false);
        return go;
    }

    private static GameObject _MakeText(string name, Transform parent, string content,
                                        int size, FontStyle style, TextAnchor anchor)
    {
        var go = _MakeRect(name, parent);
        var txt = go.AddComponent<Text>();
        txt.text = content;
        txt.fontSize = size;
        txt.fontStyle = style;
        txt.alignment = anchor;
        txt.color = Color.white;
        txt.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (txt.font == null)
            txt.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        return go;
    }

    private static Button _MakeButton(string name, Transform parent,
                                      string label, Color bgColor)
    {
        var go = _MakeRect(name, parent);
        var img = go.AddComponent<Image>();
        img.color = bgColor;

        var btn = go.AddComponent<Button>();
        var cb = btn.colors;
        cb.highlightedColor = bgColor * 1.25f;
        cb.pressedColor = bgColor * 0.75f;
        btn.colors = cb;

        var txtGO = _MakeText("Label", go.transform, label,
                              13, FontStyle.Bold, TextAnchor.MiddleCenter);
        _FillRect(txtGO);
        return btn;
    }

    // RectTransform layout helpers
    private static void _SetRect(GameObject go,
                                 Vector2 anchorMin, Vector2 anchorMax,
                                 Vector2 offsetMin, Vector2 offsetMax)
    {
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = anchorMin;
        rt.anchorMax = anchorMax;
        rt.offsetMin = offsetMin;
        rt.offsetMax = offsetMax;
    }

    private static void _FillRect(GameObject go)
    {
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = Vector2.zero;
        rt.anchorMax = Vector2.one;
        rt.offsetMin = rt.offsetMax = Vector2.zero;
    }

    private static void _PadRect(GameObject go, float left, float right, float bottom, float top)
    {
        var rt = go.GetComponent<RectTransform>();
        rt.offsetMin += new Vector2(left, bottom);
        rt.offsetMax -= new Vector2(right, top);
    }
}