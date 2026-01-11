using UnityEngine;
using System;
using System.Collections.Generic;
using System.Text;
using System.Net.Sockets; // Added for socket functionality
using System.Threading.Tasks; // Added for async operation

public class RosLidarSimulator : MonoBehaviour
{
    [Header("LIDAR Sensor Parameters")]
    [Tooltip("Number of rays per complete 360° scan")]
    public int raysPerScan = 360;

    [Tooltip("Maximum detection range in meters")]
    public float maxRange = 30.0f;

    [Tooltip("Minimum detection range in meters")]
    public float minRange = 0.1f;

    [Tooltip("Scan frequency in Hz (scans per second)")]
    public float scanFrequency = 10.0f;

    [Tooltip("Angular offset from forward direction in degrees")]
    public float angleOffset = 0.0f;

    [Header("ROS Message Configuration")]
    public string frameId = "laser_frame";
    public bool outputToConsole = false;
    public bool visualizeRays = true;

    [Header("Data Output")]
    public OutputMethod outputMethod = OutputMethod.JSONFile;
    public string outputFilePath = "lidar_scan.json";

    [Header("Network Socket Configuration")]
    public string serverIP = "127.0.0.1"; // Localhost by default
    public int serverPort = 65432;       // Must match Python server port

    // Private variables
    private float scanPeriod;
    private float timer;
    private float[] ranges;
    private float[] intensities;
    private uint sequenceNumber = 0;
    private System.Diagnostics.Stopwatch stopwatch;

    // Socket client for network communication
    private TcpClient socketClient;
    private NetworkStream networkStream;
    private bool isSocketConnected = false;

    // ROS LaserScan message structure
    [Serializable]
    public class RosHeader
    {
        public uint seq;
        public RosTime stamp;
        public string frame_id;

        public RosHeader(uint sequence, RosTime timeStamp, string frame)
        {
            seq = sequence;
            stamp = timeStamp;
            frame_id = frame;
        }
    }
    private Vector3 lastPosition;
    private float lastRotationY;

    [Serializable]
    public class RosTime
    {
        public int sec;
        public int nsec;

        public RosTime(int seconds, int nanoseconds)
        {
            sec = seconds;
            nsec = nanoseconds;
        }
    }

    [Serializable]
    public class OdometryDelta
    {
        public float dx;     // Change in X (meters)
        public float dy;     // Change in Y (meters)
        public float dtheta; // Change in Heading (degrees)
    }

    [Serializable]
    public class LaserScanMessage
    {
        public RosHeader header;
        public OdometryDelta odom; // Added odometry delta
        public float angle_min;
        public float angle_max;
        public float angle_increment;
        public float scan_time;
        public float range_min;
        public float range_max;
        public List<int> ranges_mm; // BreezySLAM prefers mm
        public List<float> intensities;

        public string ToJson()
        {
            return JsonUtility.ToJson(this);
        }
    }

    // Update this in Start()
    void Start()
    {
        InitializeLidar();
        lastPosition = transform.position;
        lastRotationY = transform.eulerAngles.y;

        if (outputMethod == OutputMethod.NetworkSocket)
        {
            InitializeNetworkSocket();
        }
    }

    public enum OutputMethod
    {
        JSONFile,
        NetworkSocket,
        ConsoleOnly
    }

    void InitializeLidar()
    {
        // Calculate derived parameters
        scanPeriod = 1.0f / scanFrequency;
        timer = 0f;

        // Initialize arrays
        ranges = new float[raysPerScan];
        intensities = new float[raysPerScan];

        // Initialize stopwatch for timestamps
        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();

        Debug.Log($"LiDAR Simulator Initialized:");
        Debug.Log($"- Rays per scan: {raysPerScan}");
        Debug.Log($"- Scan frequency: {scanFrequency} Hz");
        Debug.Log($"- Range: {minRange} to {maxRange} meters");
        Debug.Log($"- Angular resolution: {360.0f / raysPerScan:F3} degrees");
    }

    void InitializeNetworkSocket()
    {
        try
        {
            socketClient = new TcpClient();
            // Connect to the Python server asynchronously to avoid blocking
            Task connectTask = socketClient.ConnectAsync(serverIP, serverPort);

            // Use a continuation to handle the connection result
            connectTask.ContinueWith(task =>
            {
                if (task.IsFaulted)
                {
                    Debug.LogError($"Failed to connect to Python server at {serverIP}:{serverPort}. Error: {task.Exception?.Message}");
                    isSocketConnected = false;
                }
                else if (task.IsCompleted)
                {
                    networkStream = socketClient.GetStream();
                    isSocketConnected = true;
                    Debug.Log($"Successfully connected to Python server at {serverIP}:{serverPort}");
                }
            });
        }
        catch (Exception e)
        {
            Debug.LogError($"Error initializing network socket: {e.Message}");
            isSocketConnected = false;
        }
    }

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= scanPeriod)
        {
            PerformScan();
            PublishScan();
            timer = 0f;
        }
    }

    void PerformScan()
    {
        float angleStep = 360.0f / raysPerScan;

        for (int i = 0; i < raysPerScan; i++)
        {
            // Calculate current angle (starting from -180 degrees)
            float angle = -180f + (i * angleStep) + angleOffset;

            // Calculate ray direction
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;

            // Perform raycast
            RaycastHit hit;
            bool hasHit = Physics.Raycast(transform.position, direction, out hit, maxRange);

            if (hasHit && hit.distance >= minRange)
            {
                ranges[i] = Mathf.Clamp(hit.distance, minRange, maxRange);

                // Simulate intensity based on distance and angle
                intensities[i] = SimulateIntensity(hit.distance, hit.normal, direction);

                // Visualize the ray if enabled
                if (visualizeRays)
                {
                    Debug.DrawRay(transform.position, direction * hit.distance,
                                Color.Lerp(Color.red, Color.green, hit.distance / maxRange),
                                scanPeriod);
                }
            }
            else
            {
                // No valid hit within range
                ranges[i] = float.PositiveInfinity;
                intensities[i] = 0f;

                if (visualizeRays)
                {
                    Debug.DrawRay(transform.position, direction * maxRange,
                                Color.blue, scanPeriod);
                }
            }
        }
    }

    float SimulateIntensity(float distance, Vector3 surfaceNormal, Vector3 rayDirection)
    {
        // Simple intensity simulation based on:
        // 1. Distance (inverse square law approximation)
        // 2. Incidence angle (cosine of angle between ray and normal)
        // 3. Random variation

        float distanceFactor = Mathf.Clamp01(1.0f / (distance * distance));
        float cosineIncidence = Mathf.Abs(Vector3.Dot(surfaceNormal.normalized, -rayDirection.normalized));
        float randomVariation = UnityEngine.Random.Range(0.8f, 1.2f);

        return Mathf.Clamp01(distanceFactor * cosineIncidence * randomVariation);
    }

    void PublishScan()
    {
        // 1. Calculate Odometry Delta
        Vector3 currentPos = transform.position;
        float currentRotY = transform.eulerAngles.y;

        // Calculate world movement
        Vector3 worldDelta = currentPos - lastPosition;

        // Rotate world movement into the robot's local frame (at the start of the move)
        // In Unity: Z is forward, X is right
        Vector3 localDelta = Quaternion.Euler(0, -lastRotationY, 0) * worldDelta;

        // Calculate change in rotation (handling 360/0 wrap-around)
        float deltaTheta = Mathf.DeltaAngle(lastRotationY, currentRotY);

        OdometryDelta odomDelta = new OdometryDelta
        {
            dx = localDelta.z,
            dy = localDelta.x,
            dtheta = -deltaTheta    // Add the MINUS sign here
        };

        // 2. Prepare Scan Data (Convert meters to millimeters)
        List<int> rangesMM = new List<int>();

        // REVERSE the loop to convert Clockwise (Unity) to Counter-Clockwise (SLAM)
        for (int i = raysPerScan - 1; i >= 0; i--)
        {
            float r = ranges[i];
            if (float.IsPositiveInfinity(r) || r >= maxRange)
                rangesMM.Add((int)(maxRange * 1000));
            else
                rangesMM.Add((int)(r * 1000));
        }

        // 3. Construct Message
        long elapsedNanoseconds = stopwatch.ElapsedTicks * 100;
        LaserScanMessage scanMsg = new LaserScanMessage
        {
            header = new RosHeader(
                sequenceNumber++,
                new RosTime((int)(elapsedNanoseconds / 1000000000), (int)(elapsedNanoseconds % 1000000000)),
                frameId
            ),
            odom = odomDelta,
            angle_min = -Mathf.PI,
            angle_max = Mathf.PI,
            angle_increment = (2 * Mathf.PI) / raysPerScan,
            scan_time = scanPeriod,
            range_min = minRange,
            range_max = maxRange,
            ranges_mm = rangesMM,
            intensities = new List<float>(intensities)
        };

        // 4. Output
        // Output based on selected method
        switch (outputMethod)
        {
            case OutputMethod.JSONFile:
                OutputToJsonFile(scanMsg);
                break;

            case OutputMethod.ConsoleOnly:
                if (outputToConsole)
                {
                    Debug.Log(scanMsg.ToJson());
                }
                break;

            case OutputMethod.NetworkSocket:
                SendOverNetworkSocket(scanMsg);
                break;
        }
        // Update "Last" markers for next scan
        lastPosition = currentPos;
        lastRotationY = currentRotY;
    }

    void OutputToJsonFile(LaserScanMessage scanMsg)
    {
        try
        {
            string json = scanMsg.ToJson();
            System.IO.File.WriteAllText(outputFilePath, json);

            if (outputToConsole)
            {
                Debug.Log($"Scan #{sequenceNumber - 1} written to: {outputFilePath}");
                Debug.Log($"Sample ranges [0-4]: {ranges[0]:F2}, {ranges[1]:F2}, {ranges[2]:F2}, {ranges[3]:F2}, {ranges[4]:F2}");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to write scan to file: {e.Message}");
        }
    }

    void SendOverNetworkSocket(LaserScanMessage scanMsg)
    {
        if (!isSocketConnected)
        {
            if (outputToConsole)
            {
                Debug.LogWarning("Socket not connected. Attempting to reconnect...");
            }

            // Try to reconnect
            InitializeNetworkSocket();

            if (!isSocketConnected)
            {
                Debug.LogError("Cannot send scan: Socket connection failed.");
                return;
            }
        }

        try
        {
            // Convert the message to JSON
            string jsonData = scanMsg.ToJson();

            // Add a newline delimiter so Python knows where each message ends
            jsonData += "\n";

            // Convert string to bytes
            byte[] dataBytes = Encoding.UTF8.GetBytes(jsonData);

            // Send the data asynchronously
            networkStream.WriteAsync(dataBytes, 0, dataBytes.Length);

            if (outputToConsole)
            {
                Debug.Log($"Scan #{sequenceNumber - 1} sent via socket. Size: {dataBytes.Length} bytes");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to send scan over socket: {e.Message}");
            isSocketConnected = false;

            // Attempt to clean up and reconnect on next scan
            CleanupSocket();
        }
    }

    void CleanupSocket()
    {
        try
        {
            if (networkStream != null)
            {
                networkStream.Close();
                networkStream = null;
            }

            if (socketClient != null)
            {
                socketClient.Close();
                socketClient = null;
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Error during socket cleanup: {e.Message}");
        }

        isSocketConnected = false;
    }

    void OnDestroy()
    {
        // Clean up socket resources when the object is destroyed
        CleanupSocket();

        if (stopwatch != null && stopwatch.IsRunning)
        {
            stopwatch.Stop();
        }
    }

    // Public method to manually trigger a scan
    public LaserScanMessage GetCurrentScan()
    {
        // 1. Calculate movement delta for this specific moment
        Vector3 currentPos = transform.position;
        float currentRotY = transform.eulerAngles.y;

        Vector3 worldDelta = currentPos - lastPosition;
        Vector3 localDelta = Quaternion.Euler(0, -lastRotationY, 0) * worldDelta;
        float deltaTheta = Mathf.DeltaAngle(lastRotationY, currentRotY);

        OdometryDelta odomDelta = new OdometryDelta
        {
            dx = localDelta.z,
            dy = localDelta.x,
            dtheta = -deltaTheta
        };

        // 2. Perform the physical raycasts
        PerformScan();

        // 3. Create the message using the delta we just calculated
        return CreateLaserScanMessage(odomDelta);
    }

    LaserScanMessage CreateLaserScanMessage(OdometryDelta odomDelta)
    {
        // 1. Calculate Timestamps
        long elapsedNanoseconds = stopwatch.ElapsedTicks * 100;
        int seconds = (int)(elapsedNanoseconds / 1000000000);
        int nanoseconds = (int)(elapsedNanoseconds % 1000000000);

        // 2. Convert and Reverse Ranges
        // BreezySLAM/ROS expects CCW (Counter-Clockwise). 
        // Unity's default loop is CW (Clockwise).
        List<int> rangesMM = new List<int>(raysPerScan);
        List<float> intensitiesCCW = new List<float>(raysPerScan);

        for (int i = raysPerScan - 1; i >= 0; i--)
        {
            float r = ranges[i];

            // Convert to mm and handle Infinity/Max Range
            if (float.IsPositiveInfinity(r) || r >= maxRange)
            {
                rangesMM.Add((int)(maxRange * 1000));
            }
            else
            {
                rangesMM.Add((int)(r * 1000));
            }

            intensitiesCCW.Add(intensities[i]);
        }

        // 3. Return the corrected message
        return new LaserScanMessage
        {
            header = new RosHeader(
                sequenceNumber,
                new RosTime(seconds, nanoseconds),
                frameId
            ),
            odom = odomDelta, // Crucial: Attach the movement delta here
            angle_min = -Mathf.PI,
            angle_max = Mathf.PI,
            angle_increment = (2 * Mathf.PI) / raysPerScan,
            scan_time = scanPeriod,
            range_min = minRange,
            range_max = maxRange,
            ranges_mm = rangesMM, // Now correctly formatted for BreezySLAM
            intensities = intensitiesCCW
        };
    }

    // Utility method to validate sensor parameters
    void OnValidate()
    {
        raysPerScan = Mathf.Clamp(raysPerScan, 1, 1080);
        maxRange = Mathf.Max(maxRange, 0.1f);
        minRange = Mathf.Clamp(minRange, 0.01f, maxRange - 0.01f);
        scanFrequency = Mathf.Clamp(scanFrequency, 0.1f, 100f);
    }

    // Draw sensor range gizmo in editor
    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireSphere(transform.position, maxRange);
        Gizmos.DrawWireSphere(transform.position, minRange);

        // Draw forward direction
        Gizmos.color = Color.green;
        Gizmos.DrawRay(transform.position, transform.forward * maxRange);
    }
}