using UnityEngine;
using System.Collections;
using System.Net.Sockets;
using System.IO;
using System;


public class SensorSender : MonoBehaviour
{
    public int port = 5004;
    public string host = "127.0.0.1";
    public float sendInterval = 0.1f;        // 10 Hz
    public int depthWidth = 64;
    public int depthHeight = 64;
    public int jpegQuality = 75;
    public Material linearDepthMaterial;      // assign in Inspector

    private TcpClient client;
    private NetworkStream stream;
    private StreamWriter writer;
    private Camera cam;
    private RenderTexture linearDepthRT;
    private Texture2D depthTex2D;
    private Texture2D rgbTex2D;
    private float[] depthBuffer;
    private bool connected = false;
    private Rect depthRect;

    void Start()
    {
        cam = GetComponent<Camera>();
        if (cam == null)
        {
            Debug.LogError("SensorSender must be attached to a camera.");
            return;
        }

        // Enable depth texture generation
        cam.depthTextureMode = DepthTextureMode.Depth;

        // Create render texture for linear depth
        linearDepthRT = new RenderTexture(depthWidth, depthHeight, 0, RenderTextureFormat.RFloat);
        linearDepthRT.Create();

        // Texture2D to read pixels
        depthTex2D = new Texture2D(depthWidth, depthHeight, TextureFormat.RFloat, false);
        depthRect = new Rect(0, 0, depthWidth, depthHeight);
        depthBuffer = new float[depthWidth * depthHeight];

        // RGB capture texture
        rgbTex2D = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);

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
            Debug.Log($"SensorSender connected to Python on port {port}");
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

            // --- Capture linear depth ---
            Graphics.Blit(null, linearDepthRT, linearDepthMaterial);

            RenderTexture.active = linearDepthRT;
            depthTex2D.ReadPixels(depthRect, 0, 0);
            depthTex2D.Apply();
            RenderTexture.active = null;

            Color[] depthPixels = depthTex2D.GetPixels();
            for (int i = 0; i < depthPixels.Length; i++)
                depthBuffer[i] = depthPixels[i].r;

            byte[] depthBytes = new byte[depthBuffer.Length * 4];
            Buffer.BlockCopy(depthBuffer, 0, depthBytes, 0, depthBytes.Length);
            string depthBase64 = Convert.ToBase64String(depthBytes);

            // --- Capture RGB ---
            rgbTex2D.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
            rgbTex2D.Apply();
            byte[] jpgBytes = rgbTex2D.EncodeToJPG(jpegQuality);
            string rgbBase64 = Convert.ToBase64String(jpgBytes);

            // --- Get camera world pose ---
            Vector3 camPos = cam.transform.position;
            Quaternion camRot = cam.transform.rotation;

            // --- Build message ---
            SensorMessage msg = new SensorMessage
            {
                rgb = rgbBase64,
                depth = depthBase64,
                depth_width = depthWidth,
                depth_height = depthHeight,
                rgb_width = Screen.width,
                rgb_height = Screen.height,
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

            string json = JsonUtility.ToJson(msg);
            try
            {
                writer.WriteLine(json);
            }
            catch (Exception e)
            {
                Debug.LogWarning("Send failed: " + e.Message);
                connected = false;
                break;
            }
        }

        Debug.Log("SensorSender disconnected. Reconnecting...");
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
        Destroy(depthTex2D);
        Destroy(rgbTex2D);
    }
}

[System.Serializable]
public class SensorMessage
{
    public string rgb;          // base64 JPEG
    public string depth;         // base64 float32 array (linear meters)
    public int depth_width;
    public int depth_height;
    public int rgb_width;
    public int rgb_height;
    public float fov;            // camera field of view
    public float timestamp;
    // Camera world pose
    public float cam_pos_x;
    public float cam_pos_y;
    public float cam_pos_z;
    public float cam_rot_x;
    public float cam_rot_y;
    public float cam_rot_z;
    public float cam_rot_w;
}
