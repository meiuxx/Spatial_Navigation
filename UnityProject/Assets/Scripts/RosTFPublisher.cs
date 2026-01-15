using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;
using RosMessageTypes.Tf2;
using System;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using UnityEngine;

public class RosLaserStaticTF : MonoBehaviour
{
    public string tfTopic = "/tf";

    [Header("Frame Names")]
    public string parentFrame = "base_link";
    public string laserFrame = "laser_frame";

    [Header("Laser Offset in Unity (meters, degrees)")]
    public Vector3 laserPositionOffset = Vector3.zero;
    public Vector3 laserRotationOffset = Vector3.zero;

    private ROSConnection ros;
    private bool published = false;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TFMessageMsg>(tfTopic);
    }

    void Update()
    {
        // Publish once (static TF)
        if (!published)
        {
            PublishLaserTF();
            published = true;
        }
    }

    TimeMsg Now()
    {
        TimeSpan t = DateTime.UtcNow - new DateTime(1970, 1, 1);
        uint sec = (uint)t.TotalSeconds;
        uint nsec = (uint)((t.TotalSeconds - sec) * 1e9);
        return new TimeMsg(sec, nsec);
    }

    void PublishLaserTF()
    {
        var tf = new TransformStampedMsg();

        tf.header = new HeaderMsg
        {
            stamp = Now(),
            frame_id = parentFrame
        };

        tf.child_frame_id = laserFrame;

        // Unity → ROS coordinate conversion
        Vector3 rosPos = new Vector3(
            laserPositionOffset.z,
            -laserPositionOffset.x,
            laserPositionOffset.y
        );

        Quaternion rosRot =
            Quaternion.Euler(laserRotationOffset) *
            Quaternion.Euler(-90f, 0f, 0f);

        tf.transform = new TransformMsg
        {
            translation = rosPos.To<FLU>(),
            rotation = rosRot.To<FLU>()
        };

        var msg = new TFMessageMsg
        {
            transforms = new[] { tf }
        };

        ros.Publish(tfTopic, msg);

        Debug.Log("Published static TF: base_link → laser_frame");
    }
}
