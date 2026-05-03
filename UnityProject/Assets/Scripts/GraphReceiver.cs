using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using System;

using System.IO;

public class GraphVisualizer : MonoBehaviour
{
    public int listenPort = 5006;            // Port to listen on
    public float sphereRadius = 2f;
    public Color nodeColor = Color.cyan;
    public Color edgeColor = Color.yellow;

    // Data structures to hold the graph
    private List<Vector3> nodePositions = new List<Vector3>();
    private List<string> nodeLabels = new List<string>();
    private List<(int, int)> edges = new List<(int, int)>(); // indices into nodePositions

    // Threading
    private TcpListener listener;
    private Thread listenerThread;
    private Queue<Action> mainThreadActions = new Queue<Action>();
    private volatile bool running = true;

    void Start()
    {
        listenerThread = new Thread(ListenForGraph);
        listenerThread.IsBackground = true;
        listenerThread.Start();
    }

    void OnDrawGizmos()
    {
        if (nodePositions == null) return;

        // Draw nodes
        Gizmos.color = nodeColor;
        for (int i = 0; i < nodePositions.Count; i++)
        {
            Gizmos.DrawSphere(nodePositions[i], sphereRadius);
        }

        // Draw edges
        Gizmos.color = edgeColor;
        foreach (var edge in edges)
        {
            if (edge.Item1 < nodePositions.Count && edge.Item2 < nodePositions.Count)
            {
                Gizmos.DrawLine(nodePositions[edge.Item1], nodePositions[edge.Item2]);
            }
        }
    }

    private void ListenForGraph()
    {
        try
        {
            listener = new TcpListener(IPAddress.Any, listenPort);
            listener.Start();
            Debug.Log($"[GraphVisualizer] Listening on port {listenPort}");

            while (running)
            {
                TcpClient client = listener.AcceptTcpClient();
                Debug.Log("[GraphVisualizer] Python connected");
                NetworkStream stream = client.GetStream();
                StreamReader reader = new StreamReader(stream, Encoding.UTF8);

                string line;
                while ((line = reader.ReadLine()) != null && running)
                {
                    if (!string.IsNullOrEmpty(line))
                    {
                        Debug.Log($"[GraphVisualizer] Received: {line.Substring(0, Math.Min(line.Length, 200))}"); // log first 200 chars
                        string json = line;
                        lock (mainThreadActions)
                        {
                            mainThreadActions.Enqueue(() => UpdateGraphFromJson(json));
                        }
                    }
                }
                client.Close();
                Debug.Log("[GraphVisualizer] Python disconnected");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[GraphVisualizer] Server error: {e.Message}");
        }
    }

    private void UpdateGraphFromJson(string json)
    {
        // Use a simple JSON parser (Newtonsoft.Json if available, or we can use Unity's JsonUtility)
        // Since JsonUtility requires a class structure, we'll define simple ones.
        try
        {
            GraphData data = JsonUtility.FromJson<GraphData>(json);
            if (data == null || data.nodes == null) return;

            // Clear old data
            nodePositions.Clear();
            nodeLabels.Clear();
            edges.Clear();

            // Build a dictionary to map node IDs to indices
            Dictionary<string, int> idToIndex = new Dictionary<string, int>();

            // Add nodes
            for (int i = 0; i < data.nodes.Length; i++)
            {
                var node = data.nodes[i];
                nodePositions.Add(new Vector3(node.pos[0], node.pos[1], node.pos[2]));
                nodeLabels.Add(node.label);
                idToIndex[node.id] = i;
            }

            // Add edges
            if (data.edges != null)
            {
                foreach (var edge in data.edges)
                {
                    if (idToIndex.ContainsKey(edge.from) && idToIndex.ContainsKey(edge.to))
                    {
                        edges.Add((idToIndex[edge.from], idToIndex[edge.to]));
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to parse graph JSON: {e.Message}");
        }
    }

    void Update()
    {
        // Execute any queued actions on the main thread
        lock (mainThreadActions)
        {
            while (mainThreadActions.Count > 0)
            {
                mainThreadActions.Dequeue()?.Invoke();
            }
        }
    }

    void OnDestroy()
    {
        running = false;
        listenerThread?.Join(1000);
        listener?.Stop();
    }

    // --- Data structures for JSON ---
    [System.Serializable]
    public class GraphData
    {
        public GraphNode[] nodes;
        public GraphEdge[] edges;
    }

    [System.Serializable]
    public class GraphNode
    {
        public string id;
        public float[] pos;   // [x, y, z]
        public string label;
    }

    [System.Serializable]
    public class GraphEdge
    {
        public string from;
        public string to;
    }
}