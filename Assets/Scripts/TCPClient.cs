using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

[System.Serializable]
public class ObservationData
{
    public int step;
    public string[] rays;
}

public class TCPClient : MonoBehaviour
{
    [Header("Server Settings")]
    public string serverIP = "127.0.0.1";
    public int serverPort = 5001;

    private TcpClient client;
    private NetworkStream stream;
    private bool connected = false;

    void Start()
    {
        ConnectToServer();
    }

    void ConnectToServer()
    {
        try
        {
            client = new TcpClient();
            client.Connect(serverIP, serverPort);
            stream = client.GetStream();
            connected = true;
            Debug.Log("[TCPClient] Connected to Python server!");
        }
        catch (Exception e)
        {
            Debug.LogError("[TCPClient] Connection failed: " + e.Message);
        }
    }

    public void SendObservation(Texture2D frame, string[] rays, int step)
    {
        if (!connected || client == null || stream == null)
        {
            Debug.LogWarning("[TCPClient] Not connected to server, skipping send.");
            return;
        }

        try
        {
            // DEBUG: Log what we're sending
            Debug.Log($"=== TCPClient.SendObservation ===");
            Debug.Log($"Step: {step}");
            Debug.Log($"Rays array: {(rays == null ? "NULL" : $"Length: {rays.Length}")}");

            if (rays != null)
            {
                for (int i = 0; i < rays.Length; i++)
                {
                    Debug.Log($"  Ray[{i}]: {rays[i]}");
                }
            }

            // Create observation data using serializable class
            ObservationData obs = new ObservationData
            {
                step = step,
                rays = rays ?? new string[0]  // Ensure not null
            };

            string json = JsonUtility.ToJson(obs);
            Debug.Log($"JSON to send: {json}");

            byte[] jsonBytes = Encoding.UTF8.GetBytes(json);

            // Encode image as PNG
            byte[] imgBytes = frame.EncodeToPNG();
            Debug.Log($"Image bytes: {imgBytes.Length}");

            // Send lengths first (4 bytes each) - LITTLE ENDIAN
            byte[] jsonLenBytes = BitConverter.GetBytes(jsonBytes.Length);
            byte[] imgLenBytes = BitConverter.GetBytes(imgBytes.Length);

            // Ensure little-endian (Unity/Windows default)
            if (!BitConverter.IsLittleEndian)
            {
                Array.Reverse(jsonLenBytes);
                Array.Reverse(imgLenBytes);
            }

            stream.Write(jsonLenBytes, 0, 4);
            stream.Write(imgLenBytes, 0, 4);

            // Send actual data
            stream.Write(jsonBytes, 0, jsonBytes.Length);
            stream.Write(imgBytes, 0, imgBytes.Length);

            stream.Flush();
            Debug.Log($"[TCPClient] Sent step {step}, rays: {(rays == null ? 0 : rays.Length)}, image bytes: {imgBytes.Length}");
        }
        catch (Exception e)
        {
            Debug.LogError("[TCPClient] Error sending data: " + e.Message);
            connected = false;
        }
    }

    void OnApplicationQuit()
    {
        try
        {
            stream?.Close();
            client?.Close();
            connected = false;
            Debug.Log("[TCPClient] Connection closed.");
        }
        catch { }
    }
}