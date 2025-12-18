using UnityEngine;
using System.Linq;

public class SimpleFrameCollector : MonoBehaviour
{
    [Header("References")]
    public TCPClient tcpClient;
    public Camera agentCamera;
    public RGBRaycast raycastScript;

    [Header("Settings")]
    public int captureWidth = 128;
    public int captureHeight = 128;
    public int collectEveryNFrames = 30;

    private int stepCounter = 0;
    private int frameCounter = 0;

    void Update()
    {
        frameCounter++;

        if (frameCounter >= collectEveryNFrames)
        {
            CollectAndSend();
            frameCounter = 0;
        }

        // Manual override with T key
        if (Input.GetKeyDown(KeyCode.T))
        {
            CollectAndSend();
        }
    }

    void CollectAndSend()
    {
        // Perform raycast
        raycastScript.PerformDetection();

        // Get observations
        var rayObs = raycastScript.GetRayObservations();
        string[] rays = rayObs.Select(o => $"{o.objectName}:{o.distance:F2}").ToArray();

        // Capture frame
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

        // Send
        tcpClient.SendObservation(frame, rays, stepCounter);

        // Cleanup
        Destroy(frame);

        stepCounter++;

        Debug.Log($"Collected step {stepCounter} at frame {Time.frameCount}, rays: {rays.Length}");
    }
}