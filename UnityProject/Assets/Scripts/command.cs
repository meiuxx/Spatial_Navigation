using UnityEngine;
using UnityEngine.AI;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class CommandReceiver : MonoBehaviour
{
    [Header("Network")]
    public int listenPort = 5002;

    [Header("Movement")]
    public float moveSpeed = 2f;
    public float rotateSpeed = 120f;
    public float positionTolerance = 0.2f;
    public float angleTolerance = 5f;

    [Header("Optional NavMesh")]
    public bool useNavMesh = false;
    public NavMeshAgent navAgent;

    // Thread-safe goal storage
    private float pendingGoalX, pendingGoalY, pendingGoalTheta;
    private bool goalPending = false;
    private readonly object pendingLock = new object();

    // Current active goal
    private Vector3 targetPosition;
    private float targetAngle;
    private bool hasGoal = false;

    private TcpListener listener;
    private Thread listenerThread;

    void Start()
    {
        Debug.Log("[CommandReceiver] Start() called.");
        listenerThread = new Thread(ListenForCommands);
        listenerThread.IsBackground = true;
        listenerThread.Start();
    }

    void ListenForCommands()
    {
        try
        {
            listener = new TcpListener(IPAddress.Any, listenPort);
            listener.Start();
            Debug.Log($"[CommandReceiver] Listening on port {listenPort}");

            while (true)
            {
                using (var client = listener.AcceptTcpClient())
                using (var stream = client.GetStream())
                {
                    Debug.Log("[CommandReceiver] Client connected.");
                    byte[] buffer = new byte[4096];
                    int bytesRead;

                    while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        string message = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
                        Debug.Log($"[CommandReceiver] Received: {message}");
                        ProcessMessage(message);
                    }
                }
                Debug.Log("[CommandReceiver] Client disconnected.");
            }
        }
        catch (ThreadAbortException) { }
        catch (Exception e)
        {
            Debug.LogError($"[CommandReceiver] Listener error: {e.Message}");
        }
        finally
        {
            listener?.Stop();
        }
    }

    void ProcessMessage(string message)
    {
        try
        {
            MoveCommand cmd = JsonUtility.FromJson<MoveCommand>(message);
            if (cmd != null && cmd.command == "move_to")
            {
                // Store raw goal coordinates (thread-safe)
                lock (pendingLock)
                {
                    pendingGoalX = cmd.x;
                    pendingGoalY = cmd.y;
                    pendingGoalTheta = cmd.theta;
                    goalPending = true;
                }
                Debug.Log($"[CommandReceiver] Pending goal: x={cmd.x}, y={cmd.y}, theta={cmd.theta}");
            }
            else
            {
                Debug.LogWarning($"[CommandReceiver] Unknown command: {cmd?.command}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[CommandReceiver] Failed to parse message: {e.Message}\nMessage: {message}");
        }
    }

    void Update()
    {
        // Check for pending goal (main thread)
        if (goalPending)
        {
            float px, py, ptheta;
            lock (pendingLock)
            {
                px = pendingGoalX;
                py = pendingGoalY;
                ptheta = pendingGoalTheta;
                goalPending = false;
            }

            // Construct Unity target position using current Y (ground plane)
            targetPosition = new Vector3(py, transform.position.y, px);
            targetAngle = ptheta;
            hasGoal = true;
            Debug.Log($"[CommandReceiver] New goal activated: Unity position ({targetPosition.x}, {targetPosition.y}, {targetPosition.z}), angle {targetAngle}°");
        }

        if (!hasGoal)
            return;

        float distance = Vector3.Distance(transform.position, targetPosition);
        Debug.Log($"[CommandReceiver] Distance to goal: {distance:F2}m");

        if (useNavMesh && navAgent != null)
        {
            // NavMesh movement
            if (!navAgent.hasPath || navAgent.destination != targetPosition)
                navAgent.SetDestination(targetPosition);

            if (!navAgent.pathPending && navAgent.remainingDistance <= positionTolerance)
            {
                Debug.Log("[CommandReceiver] Goal reached (NavMesh).");
                hasGoal = false;
                navAgent.ResetPath();
            }
            return;
        }

        // Manual movement
        if (distance > positionTolerance)
        {
            // Direction to target (ignore vertical)
            Vector3 direction = targetPosition - transform.position;
            direction.y = 0;
            direction.Normalize();

            // Rotate towards target
            Quaternion targetRot = Quaternion.LookRotation(direction, Vector3.up);
            transform.rotation = Quaternion.RotateTowards(transform.rotation, targetRot, rotateSpeed * Time.deltaTime);

            // Move forward if facing roughly the right direction
            float angleToTarget = Vector3.Angle(transform.forward, direction);
            if (angleToTarget < 30f)
            {
                transform.Translate(Vector3.forward * moveSpeed * Time.deltaTime);
            }
        }
        else
        {
            // At target position, adjust orientation
            Quaternion desiredRot = Quaternion.Euler(0, targetAngle, 0);
            transform.rotation = Quaternion.RotateTowards(transform.rotation, desiredRot, rotateSpeed * Time.deltaTime);

            float angleDiff = Mathf.Abs(Mathf.DeltaAngle(transform.eulerAngles.y, targetAngle));
            if (angleDiff <= angleTolerance)
            {
                Debug.Log("[CommandReceiver] Goal reached.");
                hasGoal = false;
            }
        }
    }

    void OnDestroy()
    {
        listenerThread?.Abort();
        listener?.Stop();
    }
}


[Serializable]
public class MoveCommand
{
    public string command;
    public float x;      // Python forward
    public float y;      // Python right
    public float theta;  // orientation (degrees)
}
