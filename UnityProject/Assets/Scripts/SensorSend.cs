using UnityEngine;
using System.Collections;
using System.Net.Sockets;
using System.IO;
using System;

public class SensorSender : MonoBehaviour
{
    public int port = 5004;
    public string host = "127.0.0.1";
    public float sendInterval = 0.1f;
    public int depthWidth = 64;
    public int depthHeight = 64;
    public int jpegQuality = 75;
    public Material linearDepthMaterial;

    private TcpClient client;
    private NetworkStream stream;
    private StreamWriter writer;
    private Camera cam;

    private RenderTexture linearDepthRT;
    private RenderTexture rgbRT;          // NEW: camera renders here, not to screen

    private Texture2D depthTex2D;
    private Texture2D rgbTex2D;
    private float[] depthBuffer;
    private bool connected = false;

    void Start()
    {
        cam = GetComponent<Camera>();
        if (cam == null) { Debug.LogError("SensorSender must be on a Camera."); return; }

        cam.depthTextureMode = DepthTextureMode.Depth;

        // Depth RT — unchanged
        linearDepthRT = new RenderTexture(depthWidth, depthHeight, 0, RenderTextureFormat.RFloat);
        linearDepthRT.Create();
        depthTex2D = new Texture2D(depthWidth, depthHeight, TextureFormat.RFloat, false);
        depthBuffer = new float[depthWidth * depthHeight];

        // RGB RT — camera renders into this; UI overlay never touches it
        // FIX: was ReadPixels(screen) after WaitForEndOfFrame, which captured
        // the Screen Space Overlay canvas after it had already been composited.
        // Rendering into a private RenderTexture bypasses the overlay entirely.
        int w = Screen.width, h = Screen.height;
        rgbRT = new RenderTexture(w, h, 24, RenderTextureFormat.ARGB32);
        rgbRT.Create();
        rgbTex2D = new Texture2D(w, h, TextureFormat.RGB24, false);

        ConnectToServer();
    }

    void ConnectToServer()
    {
        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            stream = client.GetStream();
            writer = new StreamWriter(stream) { AutoFlush = true };
            connected = true;
            Debug.Log($"SensorSender connected on port {port}");
            StartCoroutine(SendLoop());
        }
        catch (Exception e)
        {
            Debug.LogError("Connection failed: " + e.Message);
            Invoke(nameof(ConnectToServer), 2.0f);
        }
    }

    IEnumerator SendLoop()
    {
        while (connected && client != null && client.Connected)
        {
            yield return new WaitForSeconds(sendInterval);
            yield return new WaitForEndOfFrame();

            // ── Depth (unchanged) ─────────────────────────────────────────────
            Graphics.Blit(null, linearDepthRT, linearDepthMaterial);
            RenderTexture.active = linearDepthRT;
            depthTex2D.ReadPixels(new Rect(0, 0, depthWidth, depthHeight), 0, 0);
            depthTex2D.Apply();
            RenderTexture.active = null;

            Color[] depthPixels = depthTex2D.GetPixels();
            for (int i = 0; i < depthPixels.Length; i++)
                depthBuffer[i] = depthPixels[i].r;
            byte[] depthBytes = new byte[depthBuffer.Length * 4];
            Buffer.BlockCopy(depthBuffer, 0, depthBytes, 0, depthBytes.Length);
            string depthBase64 = Convert.ToBase64String(depthBytes);

            // ── RGB — render camera into private RT, read from that ───────────
            // The camera's targetTexture is temporarily set to rgbRT so it
            // re-renders the scene geometry into an off-screen buffer.
            // Screen Space Overlay canvases are composited onto the screen
            // AFTER all camera rendering, so they are never written into rgbRT.
            var prevTarget = cam.targetTexture;
            cam.targetTexture = rgbRT;
            cam.Render();
            cam.targetTexture = prevTarget;

            RenderTexture.active = rgbRT;
            rgbTex2D.ReadPixels(new Rect(0, 0, rgbRT.width, rgbRT.height), 0, 0);
            rgbTex2D.Apply();
            RenderTexture.active = null;

            byte[] jpgBytes = rgbTex2D.EncodeToJPG(jpegQuality);
            string rgbBase64 = Convert.ToBase64String(jpgBytes);

            // ── Pose ──────────────────────────────────────────────────────────
            Vector3 camPos = cam.transform.position;
            Quaternion camRot = cam.transform.rotation;

            // ── Send ──────────────────────────────────────────────────────────
            SensorMessage msg = new SensorMessage
            {
                rgb = rgbBase64,
                depth = depthBase64,
                depth_width = depthWidth,
                depth_height = depthHeight,
                rgb_width = rgbRT.width,
                rgb_height = rgbRT.height,
                fov = cam.fieldOfView,
                timestamp = Time.time,
                cam_pos_x = camPos.x,
                cam_pos_y = camPos.y,
                cam_pos_z = camPos.z,
                cam_rot_x = camRot.x,
                cam_rot_y = camRot.y,
                cam_rot_z = camRot.z,
                cam_rot_w = camRot.w
            };

            try { writer.WriteLine(JsonUtility.ToJson(msg)); }
            catch (Exception e)
            {
                Debug.LogWarning("Send failed: " + e.Message);
                connected = false;
                break;
            }
        }

        Debug.Log("SensorSender disconnected — reconnecting…");
        CloseConnection();
        Invoke(nameof(ConnectToServer), 2.0f);
    }

    void CloseConnection()
    {
        if (writer != null) writer.Close();
        if (stream != null) stream.Close();
        if (client != null) client.Close();
        connected = false;
    }

    void OnDestroy()
    {
        CloseConnection();
        if (linearDepthRT != null) linearDepthRT.Release();
        if (rgbRT != null) rgbRT.Release();
        Destroy(depthTex2D);
        Destroy(rgbTex2D);
    }
}

[System.Serializable]
public class SensorMessage
{
    public string rgb;
    public string depth;
    public int depth_width;
    public int depth_height;
    public int rgb_width;
    public int rgb_height;
    public float fov;
    public float timestamp;
    public float cam_pos_x, cam_pos_y, cam_pos_z;
    public float cam_rot_x, cam_rot_y, cam_rot_z, cam_rot_w;
}