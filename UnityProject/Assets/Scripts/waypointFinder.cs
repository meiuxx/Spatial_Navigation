using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.AI;

public class WaypointReceiver : MonoBehaviour
{
    [Header("Network")]
    public int commandPort = 5002;      // Port for movement commands (server)
    public int imagePort = 5003;        // Port for streaming images (client)

    [Header("Cameras")]
    public Camera frontCamera;
    public Camera leftCamera;
    public Camera rightCamera;

    public int captureIntervalMs = 200;  // 5 Hz
    public int imageWidth = 256;
    public int imageHeight = 256;

    [Header("Navigation")]
    public NavMeshAgent agent;
    public float waypointPauseSeconds = 3f;
    public bool rotateAtWaypoint = true;

    // Image client fields
    private TcpClient imageClient;
    private NetworkStream imageStream;
    private Thread imageSendThread;
    private readonly object imageClientLock = new object(); // For thread‑safe reconnect

    // Command server fields
    private TcpListener commandListener;
    private Thread commandThread;

    private volatile bool isRunning = true;
    private ConcurrentQueue<Action> mainThreadActions = new ConcurrentQueue<Action>();

    void Start()
    {
        if (agent == null) agent = GetComponent<NavMeshAgent>();
        if (agent == null)
        {
            Debug.LogError("No NavMeshAgent found! Please attach one.");
            return;
        }

        // Ensure agent is on NavMesh
        if (!agent.isOnNavMesh)
        {
            NavMeshHit hit;
            if (NavMesh.SamplePosition(transform.position, out hit, 10f, NavMesh.AllAreas))
                agent.Warp(hit.position);
            else
                Debug.LogError("Cannot find NavMesh near agent.");
        }

        // Start command listener thread (remains a server)
        commandThread = new Thread(CommandServer);
        commandThread.IsBackground = true;
        commandThread.Start();

        // Connect to Python image receiver (client mode)
        ConnectToImageReceiver();

        // Start image sending thread
        imageSendThread = new Thread(ImageSendLoop);
        imageSendThread.IsBackground = true;
        imageSendThread.Start();

        Debug.Log($"WaypointReceiver started: command port {commandPort}, image port {imagePort}");
    }

    void OnDestroy()
    {
        isRunning = false;
        commandListener?.Stop();
        lock (imageClientLock)
        {
            imageClient?.Close();
            imageClient = null;
            imageStream = null;
        }
        commandThread?.Join();
        imageSendThread?.Join();
    }

    void Update()
    {
        // Execute actions queued from other threads (including image capture)
        while (mainThreadActions.TryDequeue(out var action))
            action.Invoke();
    }

    // ---------- Command Server (unchanged, remains a listener) ----------
    private void CommandServer()
    {
        commandListener = new TcpListener(IPAddress.Any, commandPort);
        commandListener.Start();
        while (isRunning)
        {
            try
            {
                var client = commandListener.AcceptTcpClient();
                var clientThread = new Thread(HandleCommandClient);
                clientThread.IsBackground = true;
                clientThread.Start(client);
            }
            catch { break; }
        }
    }

    private void HandleCommandClient(object obj)
    {
        TcpClient client = (TcpClient)obj;
        NetworkStream stream = client.GetStream();
        byte[] buffer = new byte[1024];
        StringBuilder sb = new StringBuilder();

        while (isRunning && client.Connected)
        {
            int bytesRead = 0;
            try { bytesRead = stream.Read(buffer, 0, buffer.Length); }
            catch { break; }
            if (bytesRead == 0) break;

            string data = Encoding.UTF8.GetString(buffer, 0, bytesRead);
            sb.Append(data);

            string current = sb.ToString();
            int newlineIdx;
            while ((newlineIdx = current.IndexOf('\n')) >= 0)
            {
                string line = current.Substring(0, newlineIdx).Trim();
                if (line.Length > 0)
                {
                    string cmd = line;
                    mainThreadActions.Enqueue(() => ProcessCommand(cmd, stream));
                }
                sb.Remove(0, newlineIdx + 1);
                current = sb.ToString();
            }
        }
        client.Close();
    }

    private void ProcessCommand(string command, NetworkStream stream)
    {
        string[] parts = command.Split(' ');
        if (parts.Length >= 3 && parts[0].ToUpper() == "MOVE")
        {
            if (float.TryParse(parts[1], out float x) && float.TryParse(parts[2], out float z))
            {
                Vector3 dest = new Vector3(x, transform.position.y, z);
                agent.SetDestination(dest);
                Debug.Log($"Moving to ({x}, {z})");
                SendResponse(stream, "OK");
                StartCoroutine(WaitForArrival(stream, dest));
            }
            else
                SendResponse(stream, "ERROR: Invalid coordinates");
        }
        else
            SendResponse(stream, "ERROR: Unknown command");
    }

    private IEnumerator WaitForArrival(NetworkStream stream, Vector3 destination)
    {
        yield return null;

        float timeout = 30f;
        float elapsed = 0f;

        while (elapsed < timeout)
        {
            if (!agent.isOnNavMesh)
            {
                SendResponse(stream, "ERROR: Off NavMesh");
                yield break;
            }

            if (!agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
            {
                Debug.Log("Reached waypoint, pausing for analysis...");
                yield return StartCoroutine(PauseAtWaypoint());
                SendResponse(stream, "DONE");
                yield break;
            }

            elapsed += Time.deltaTime;
            yield return null;
        }
        SendResponse(stream, "ERROR: Timeout");
    }

    private IEnumerator PauseAtWaypoint()
    {
        if (rotateAtWaypoint)
        {
            float elapsed = 0f;
            Quaternion startRot = transform.rotation;
            while (elapsed < waypointPauseSeconds)
            {
                float angle = (elapsed / waypointPauseSeconds) * 360f;
                transform.rotation = startRot * Quaternion.Euler(0, angle, 0);
                elapsed += Time.deltaTime;
                yield return null;
            }
            transform.rotation = startRot;
        }
        else
        {
            yield return new WaitForSeconds(waypointPauseSeconds);
        }
    }

    private void SendResponse(NetworkStream stream, string msg)
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(msg + "\n");
            stream.Write(data, 0, data.Length);
        }
        catch { }
    }

    // ---------- Image Client (connects to Python, sends frames) ----------
    private void ConnectToImageReceiver()
    {
        try
        {
            var client = new TcpClient();
            client.Connect("127.0.0.1", imagePort);  // Python's IP (adjust if needed)
            lock (imageClientLock)
            {
                imageClient = client;
                imageStream = client.GetStream();
            }
            Debug.Log("Connected to Python image receiver.");
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to connect to image receiver: " + e.Message);
        }
    }

    private void ImageSendLoop()
    {
        while (isRunning)
        {
            // Check if we are connected; if not, try to reconnect
            bool connected = false;
            lock (imageClientLock)
            {
                connected = imageClient != null && imageClient.Connected;
            }

            if (!connected)
            {
                Debug.LogWarning("Image client disconnected. Attempting to reconnect...");
                ConnectToImageReceiver();  // This runs on background thread – safe for socket ops
                Thread.Sleep(1000);        // Wait a bit before next attempt
                continue;
            }

            // Enqueue a capture action to be executed on the main thread
            mainThreadActions.Enqueue(CaptureAndSendImages);

            // Wait for the next capture interval
            Thread.Sleep(captureIntervalMs);
        }
    }

    private void CaptureAndSendImages()
    {
        // This runs on the main thread via the queue
        if (imageStream == null) return;

        Vector3 pos = transform.position;
        float yaw = transform.eulerAngles.y;

        Camera[] cameras = { frontCamera, leftCamera, rightCamera };
        for (int i = 0; i < cameras.Length; i++)
        {
            if (cameras[i] == null) continue;

            // Render camera to temporary texture
            RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
            cameras[i].targetTexture = rt;
            cameras[i].Render();
            RenderTexture.active = rt;

            Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();

            // Cleanup
            cameras[i].targetTexture = null;
            RenderTexture.active = null;
            Destroy(rt);

            // Encode to JPG
            byte[] jpgData = tex.EncodeToJPG();
            Destroy(tex);

            // Build header: cameraId (int), x (float), y (float), yaw (float), imageSize (int)
            byte[] header = new byte[20];
            int offset = 0;
            WriteInt(header, i, ref offset);
            WriteFloat(header, pos.x, ref offset);
            WriteFloat(header, pos.y, ref offset);
            WriteFloat(header, yaw, ref offset);
            WriteInt(header, jpgData.Length, ref offset);

            try
            {
                imageStream.Write(header, 0, header.Length);
                imageStream.Write(jpgData, 0, jpgData.Length);
                imageStream.Flush();
            }
            catch (Exception e)
            {
                Debug.LogError("Image send error: " + e.Message);
                // Mark the client as disconnected; the background thread will reconnect
                lock (imageClientLock)
                {
                    imageClient?.Close();
                    imageClient = null;
                    imageStream = null;
                }
                break; // Stop sending remaining cameras for this frame
            }
        }
    }

    private void WriteInt(byte[] buffer, int value, ref int offset)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        Array.Copy(bytes, 0, buffer, offset, 4);
        offset += 4;
    }

    private void WriteFloat(byte[] buffer, float value, ref int offset)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        Array.Copy(bytes, 0, buffer, offset, 4);
        offset += 4;
    }
}