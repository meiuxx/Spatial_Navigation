using UnityEngine;

public class MovementDetector : MonoBehaviour
{
    public RGBRaycast raycastScript; // Reference to your RGBRaycast

    [Header("Settings")]
    public float positionThreshold = 0.05f;
    public float rotationThreshold = 1f;

    private Vector3 lastPosition;
    private Quaternion lastRotation;

    public ObservationCollector collector; 

    void Start()
    {
        lastPosition = transform.position;
        lastRotation = transform.rotation;
    }

    void Update()
    {
        float positionChange = Vector3.Distance(transform.position, lastPosition);
        float rotationChange = Quaternion.Angle(transform.rotation, lastRotation);

        if (positionChange > positionThreshold || rotationChange > rotationThreshold)
        {
            collector.CollectAndSend();

            if (raycastScript != null)
            {
                raycastScript.PerformDetection();
            }

            lastPosition = transform.position;
            lastRotation = transform.rotation;
        }
    }
}