using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using System;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using UnityEngine;

public class RosLidarSimulator : MonoBehaviour
{
    [Header("LIDAR Sensor Parameters")]
    public int raysPerScan = 360;
    public float maxRange = 30.0f;
    public float minRange = 0.1f;
    public float scanFrequency = 10.0f;
    public float angleOffset = 0.0f;

    [Header("ROS Configuration")]
    public string scanTopic = "/scan";
    public string odomTopic = "/odom";
    public string tfTopic = "/tf";
    public string frameId = "laser_frame";
    public string odomFrameId = "odom";
    public string baseFrameId = "base_link";
    public string mapFrameId = "map";
    public bool visualizeRays = true;
    public GameObject robotBase;

    // Private variables
    private ROSConnection ros;
    private float scanPeriod;
    private float timer;
    private float[] ranges;
    private float[] intensities;
    private uint scanSequence = 0;
    private uint odomSequence = 0;
    private Vector3 lastPosition;
    private Quaternion lastRotation;

    private Vector3 odomPosition = Vector3.zero;
    private Quaternion odomRotation = Quaternion.identity;
    private float lastOdomTime;

    void Start()
    {
        lastOdomTime = Time.time;

        InitializeLidar();
        InitializeROS();

        if (robotBase != null)
        {
            lastPosition = robotBase.transform.position;
            lastRotation = robotBase.transform.rotation;
        }
        else
        {
            lastPosition = transform.position;
            lastRotation = transform.rotation;
        }
    }

    void PublishOdomTF(TimeMsg time)
    {
        if (robotBase == null) return;

        var tfMsg = new RosMessageTypes.Tf2.TFMessageMsg();
        var t = new RosMessageTypes.Geometry.TransformStampedMsg();

        t.header = new HeaderMsg
        {
            stamp = time,
            frame_id = "odom"
        };

        t.child_frame_id = "base_link";

        t.transform = new TransformMsg
        {
            translation = robotBase.transform.position.To<FLU>(),
            rotation = robotBase.transform.rotation.To<FLU>()
        };

        tfMsg.transforms = new[] { t };

        ros.Publish("/tf", tfMsg);
    }


    TimeMsg CreateROSTime()
    {
        // Get current time in ROS format (seconds since Unix epoch)
        TimeSpan timeSpan = DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        double totalSeconds = timeSpan.TotalSeconds;
        uint seconds = (uint)totalSeconds;
        uint nanoseconds = (uint)((totalSeconds - seconds) * 1e9);

        return new TimeMsg(seconds, nanoseconds);
    }
    void InitializeLidar()
    {
        scanPeriod = 1.0f / scanFrequency;
        timer = 0f;
        ranges = new float[raysPerScan];
        intensities = new float[raysPerScan];
        Debug.Log($"LiDAR Initialized: {raysPerScan} rays, {scanFrequency}Hz, {minRange}-{maxRange}m range");
    }

    void InitializeROS()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<LaserScanMsg>(scanTopic);
        ros.RegisterPublisher<OdometryMsg>(odomTopic);
        ros.RegisterPublisher<RosMessageTypes.Tf2.TFMessageMsg>("/tf");
        Debug.Log($"ROS Publishers: {scanTopic}, {odomTopic}");
    }

    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= scanPeriod)
        {
            PerformScan();
            PublishScanAndOdom();
            timer = 0f;
        }
    }

    void PerformScan()
    {
        float angleStep = 360.0f / raysPerScan;

        for (int i = 0; i < raysPerScan; i++)
        {
            // Calculate angle (ROS convention: 0 = forward, CCW positive)
            float angle = (i * angleStep) + angleOffset;
            Vector3 direction = Quaternion.Euler(0, -angle, 0) * transform.forward;

            RaycastHit hit;
            bool hasHit = Physics.Raycast(transform.position, direction, out hit, maxRange);

            if (hasHit && hit.distance >= minRange)
            {
                ranges[i] = Mathf.Clamp(hit.distance, minRange, maxRange);
                intensities[i] = SimulateIntensity(hit.distance, hit.normal, direction);

                if (visualizeRays)
                {
                    Debug.DrawRay(transform.position, direction * hit.distance,
                                Color.Lerp(Color.red, Color.green, hit.distance / maxRange),
                                scanPeriod);
                }
            }
            else
            {
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
        float distanceFactor = Mathf.Clamp01(1.0f / (distance * distance));
        float cosineIncidence = Mathf.Abs(Vector3.Dot(surfaceNormal.normalized, -rayDirection.normalized));
        float randomVariation = UnityEngine.Random.Range(0.8f, 1.2f);
        return Mathf.Clamp01(distanceFactor * cosineIncidence * randomVariation);
    }

    void PublishScanAndOdom()
    {
        // Get current ROS time
        var time = CreateROSTime();

        // Publish both messages
        PublishLaserScan(time);
        PublishOdometry(time);
        PublishOdomTF(time);


        // Update last position/rotation
        if (robotBase != null)
        {
            lastPosition = robotBase.transform.position;
            lastRotation = robotBase.transform.rotation;
        }
    }

    void PublishLaserScan(TimeMsg time)
    {
        var scanMsg = new LaserScanMsg();

        // CRITICAL FIX 1: Proper header with incrementing sequence
        scanMsg.header = new HeaderMsg
        {
            seq = scanSequence++,
            stamp = time,
            frame_id = frameId
        };

        // CRITICAL FIX 2: Correct angle range (-π to π)
        scanMsg.angle_min = -Mathf.PI;  // -180 degrees
        scanMsg.angle_max = Mathf.PI;   // 180 degrees
        scanMsg.angle_increment = (2f * Mathf.PI) / raysPerScan;
        scanMsg.time_increment = 0f;
        scanMsg.scan_time = scanPeriod;
        scanMsg.range_min = minRange;
        scanMsg.range_max = maxRange;

        // Fill ranges array
        scanMsg.ranges = new float[raysPerScan];
        scanMsg.intensities = new float[raysPerScan];

        for (int i = 0; i < raysPerScan; i++)
        {
            if (float.IsPositiveInfinity(ranges[i]) || ranges[i] > maxRange)
            {
                scanMsg.ranges[i] = 0f;  // ROS uses 0 for no return
            }
            else
            {
                scanMsg.ranges[i] = ranges[i];
            }
            scanMsg.intensities[i] = intensities[i];
        }

        // Publish the scan
        ros.Publish(scanTopic, scanMsg);

        // Debug log every 10 scans
        if (scanSequence % 10 == 0)
        {
            Debug.Log($"Published scan #{scanSequence} with {raysPerScan} points");
        }
    }


    void PublishOdometry(TimeMsg time)
    {
        if (robotBase == null) return;

        // Time delta
        float dt = Time.time - lastOdomTime;
        if (dt <= 0f) return;
        lastOdomTime = Time.time;

        // Current pose in Unity
        Vector3 currentPos = robotBase.transform.position;
        Quaternion currentRot = robotBase.transform.rotation;

        // Delta motion in Unity world
        Vector3 deltaPos = currentPos - lastPosition;
        Quaternion deltaRot = currentRot * Quaternion.Inverse(lastRotation);

        // Integrate odometry (LOCAL FRAME)
        odomPosition += odomRotation * deltaPos;
        odomRotation = odomRotation * deltaRot;

        // Linear velocity in base frame
        Vector3 linearVel = deltaPos / dt;
        Vector3 angularVel = deltaRot.eulerAngles * Mathf.Deg2Rad / dt;

        var odomMsg = new OdometryMsg
        {
            header = new HeaderMsg
            {
                seq = odomSequence++,
                stamp = time,
                frame_id = odomFrameId
            },
            child_frame_id = baseFrameId,
            pose = new PoseWithCovarianceMsg
            {
                pose = new PoseMsg
                {
                    position = odomPosition.To<FLU>(),
                    orientation = odomRotation.To<FLU>()
                }
            },
            twist = new TwistWithCovarianceMsg
            {
                twist = new TwistMsg
                {
                    linear = linearVel.To<FLU>(),
                    angular = angularVel.To<FLU>()
                }
            }
        };

        ros.Publish(odomTopic, odomMsg);

        lastPosition = currentPos;
        lastRotation = currentRot;
    }



    void OnValidate()
    {
        raysPerScan = Mathf.Clamp(raysPerScan, 36, 1080); // Minimum 36 for SLAM
        maxRange = Mathf.Max(maxRange, 1.0f);
        minRange = Mathf.Clamp(minRange, 0.05f, maxRange * 0.5f);
        scanFrequency = Mathf.Clamp(scanFrequency, 1.0f, 30.0f); // 1-30Hz for SLAM
    }
}