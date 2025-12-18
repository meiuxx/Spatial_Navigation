using UnityEngine;
using System.Linq;

public class ObservationCollector : MonoBehaviour
{
    [Header("References")]
    public TCPClient tcpClient;
    public Camera agentCamera;
    public RGBRaycast raycastScript;

    [Header("Capture Settings")]
    public int captureWidth = 128;
    public int captureHeight = 128;

    private int stepCounter = 0;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.T))
        {
            CollectAndSend();
        }

        // Optional: Also collect on movement if you want
        if (Input.GetKeyDown(KeyCode.M))
        {
            raycastScript.PerformDetection();
            Debug.Log($"Manual raycast: {raycastScript.GetRayObservations().Count} objects");
        }
    }

    public void CollectAndSend()
    {
        Debug.Log($"=== COLLECTING OBSERVATION {stepCounter} ===");

        if (tcpClient == null || agentCamera == null || raycastScript == null)
        {
            Debug.LogWarning("[ObservationCollector] Missing references!");
            return;
        }

        // --- Force raycast update ---
        raycastScript.PerformDetection();

        // --- Get ray observations ---
        var rayObs = raycastScript.GetRayObservations();
        string[] rays = rayObs.Select(o => $"{o.objectName}:{o.distance:F2}").ToArray();

        Debug.Log($"Ray observations count: {rayObs.Count}");
        if (rayObs.Count > 0)
        {
            foreach (var obs in rayObs.Take(3))
            {
                Debug.Log($"  - {obs.objectName} at {obs.distance:F2}m");
            }
            if (rayObs.Count > 3)
            {
                Debug.Log($"  ... and {rayObs.Count - 3} more");
            }
        }

        // --- Capture RGB frame ---
        RenderTexture rt = new RenderTexture(captureWidth, captureHeight, 24);
        agentCamera.targetTexture = rt;
        Texture2D frame = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);

        agentCamera.Render();
        RenderTexture.active = rt;
        frame.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        frame.Apply();

        agentCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        Debug.Log($"Captured frame: {frame.width}x{frame.height}");

        // --- Send via TCP ---
        tcpClient.SendObservation(frame, rays, stepCounter);

        Debug.Log($"[ObservationCollector] Sent step {stepCounter} with {rays.Length} rays");

        // Cleanup
        Destroy(frame);

        stepCounter++;
    }
}