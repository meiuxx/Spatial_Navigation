using UnityEngine;
using System;
using System.IO;
using System.Collections;

// ─────────────────────────────────────────────────────────────────────────────
// ScreenshotCapture.cs
//
// Attach to any GameObject in the scene (does not need to be a camera).
// Press P to capture the current frame and save it as a JPEG to disk.
//
// Saved to:  Application.persistentDataPath/Screenshots/
// Filename:  screenshot_YYYYMMDD_HHmmss_fff.jpg
//
// No networking. No SensorSender. Completely standalone.
// ─────────────────────────────────────────────────────────────────────────────

public class ScreenshotCapture : MonoBehaviour
{
    [Header("Capture")]
    public KeyCode captureKey = KeyCode.P;
    public int jpegQuality = 95;

    [Header("Save location")]
    [Tooltip("Subfolder inside Application.persistentDataPath")]
    public string saveFolder = "Screenshots";

    private string savePath;
    private bool captureQueued = false;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void Start()
    {
        savePath = Path.Combine(Application.persistentDataPath, saveFolder);
        Directory.CreateDirectory(savePath);
        Debug.Log($"[ScreenshotCapture] Saving to: {savePath}");
        Debug.Log($"[ScreenshotCapture] Press [{captureKey}] to capture.");
        StartCoroutine(CaptureLoop());
    }

    void Update()
    {
        if (Input.GetKeyDown(captureKey))
        {
            captureQueued = true;
            Debug.Log("[ScreenshotCapture] Capture queued...");
        }
    }

    void OnDestroy()
    {
        StopAllCoroutines();
    }

    // ── Capture coroutine ─────────────────────────────────────────────────────

    IEnumerator CaptureLoop()
    {
        while (true)
        {
            yield return new WaitUntil(() => captureQueued);
            captureQueued = false;

            // Must wait for end of frame before reading pixels
            yield return new WaitForEndOfFrame();

            Texture2D tex = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);

            try
            {
                tex.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
                tex.Apply();

                byte[] jpg = tex.EncodeToJPG(jpegQuality);
                string filename = $"screenshot_{DateTime.Now:yyyyMMdd_HHmmss_fff}.jpg";
                string fullPath = Path.Combine(savePath, filename);

                File.WriteAllBytes(fullPath, jpg);
                Debug.Log($"[ScreenshotCapture] Saved → {fullPath}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[ScreenshotCapture] Failed to save: {e.Message}");
            }
            finally
            {
                Destroy(tex);
            }
        }
    }
}