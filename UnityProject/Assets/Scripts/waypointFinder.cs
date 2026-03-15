using UnityEngine;
using UnityEngine.AI;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Concurrent;
using System.Collections;

public class WaypointReceiver : MonoBehaviour
{
    public int port = 5002;
    public NavMeshAgent agent;

    private TcpListener listener;
    private Thread listenerThread;
    private volatile bool isRunning = true;
    private ConcurrentQueue<System.Action> mainThreadActions = new ConcurrentQueue<System.Action>();

    void Start()
    {
        if (agent == null)
            agent = GetComponent<NavMeshAgent>();

        if (agent == null)
        {
            Debug.LogError("No NavMeshAgent found! Please attach one to this GameObject.");
            return;
        }

        agent.enabled = true;

        // Ensure the agent is on the NavMesh
        if (!agent.isOnNavMesh)
        {
            Debug.LogWarning("Agent is not on the NavMesh. Attempting to warp to nearest point.");
            NavMeshHit hit;
            if (NavMesh.SamplePosition(transform.position, out hit, 10f, NavMesh.AllAreas))
            {
                agent.Warp(hit.position);
                Debug.Log($"Agent warped to {hit.position}");
            }
            else
            {
                Debug.LogError("Cannot find a valid NavMesh near the agent. Make sure the floor is baked.");
            }
        }

        listenerThread = new Thread(ListenForClients);
        listenerThread.IsBackground = true;
        listenerThread.Start();
        Debug.Log($"WaypointReceiver started on port {port}");
    }

    void OnDestroy()
    {
        isRunning = false;
        listener?.Stop();
        listenerThread?.Join();
    }

    void Update()
    {
        while (mainThreadActions.TryDequeue(out var action))
            action.Invoke();
    }

    private void ListenForClients()
    {
        try
        {
            listener = new TcpListener(IPAddress.Any, port);
            listener.Start();
            Debug.Log($"Listening on port {port}...");

            while (isRunning)
            {
                var client = listener.AcceptTcpClient();
                Debug.Log("Client connected!");
                var clientThread = new Thread(HandleClient);
                clientThread.IsBackground = true;
                clientThread.Start(client);
            }
        }
        catch (SocketException ex)
        {
            if (isRunning)
                Debug.LogError($"Socket error: {ex.Message}");
        }
        finally
        {
            listener?.Stop();
        }
    }

    private void HandleClient(object obj)
    {
        TcpClient client = (TcpClient)obj;
        NetworkStream stream = client.GetStream();
        byte[] buffer = new byte[1024];
        StringBuilder sb = new StringBuilder();

        while (isRunning && client.Connected)
        {
            int bytesRead;
            try
            {
                bytesRead = stream.Read(buffer, 0, buffer.Length);
            }
            catch
            {
                break;
            }

            if (bytesRead == 0) break;

            string data = Encoding.UTF8.GetString(buffer, 0, bytesRead);
            sb.Append(data);

            string current = sb.ToString();
            int newlineIndex;
            while ((newlineIndex = current.IndexOf('\n')) >= 0)
            {
                string line = current.Substring(0, newlineIndex).Trim();
                if (line.Length > 0)
                {
                    mainThreadActions.Enqueue(() => ProcessCommand(line, stream));
                }
                sb.Remove(0, newlineIndex + 1);
                current = sb.ToString();
            }
        }

        client.Close();
        Debug.Log("Client disconnected.");
    }

    private void ProcessCommand(string command, NetworkStream stream)
    {
        string[] parts = command.Split(' ');
        if (parts.Length >= 3 && parts[0].ToUpper() == "MOVE")
        {
            if (float.TryParse(parts[1], out float x) && float.TryParse(parts[2], out float z))
            {
                Vector3 destination = new Vector3(x, transform.position.y, z);
                agent.SetDestination(destination);
                Debug.Log($"Moving to ({x}, {z})");

                SendResponse(stream, "OK");
                StartCoroutine(WaitForArrival(stream, destination));
            }
            else
            {
                Debug.LogWarning($"Invalid coordinates: {command}");
                SendResponse(stream, "ERROR: Invalid coordinates");
            }
        }
        else
        {
            Debug.LogWarning($"Unknown command: {command}");
            SendResponse(stream, "ERROR: Unknown command");
        }
    }

    private IEnumerator WaitForArrival(NetworkStream stream, Vector3 destination)
    {
        // Wait a frame for the path to start computing
        yield return null;

        // Check if agent is on NavMesh and has a path
        float timeout = 20f; // seconds
        float elapsed = 0f;

        while (elapsed < timeout)
        {
            if (!agent.isOnNavMesh)
            {
                Debug.LogError("Agent left the NavMesh! Aborting.");
                SendResponse(stream, "ERROR: Off NavMesh");
                yield break;
            }

            if (!agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
            {
                // Reached destination
                Debug.Log("Reached destination, sending DONE");
                SendResponse(stream, "DONE");
                yield break;
            }

            elapsed += Time.deltaTime;
            yield return null;
        }

        // Timeout
        Debug.LogWarning("Move timeout reached");
        SendResponse(stream, "ERROR: Timeout");
    }

    private void SendResponse(NetworkStream stream, string message)
    {
        try
        {
            byte[] response = Encoding.UTF8.GetBytes(message + "\n");
            stream.Write(response, 0, response.Length);
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Failed to send response: {ex.Message}");
        }
    }
}