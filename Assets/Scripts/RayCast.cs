using UnityEngine;
using System.Collections.Generic;

public class RGBRaycast : MonoBehaviour
{
    public Camera rgbCamera;

    [Header("Detection Settings")]
    public float maxDistance = 100f;
    public LayerMask raycastMask = -1;
    public int rayCount = 9;

    [Header("Update Settings")]
    public UpdateMode updateMode = UpdateMode.OnMovement;
    public float positionThreshold = 0.1f;
    public float rotationThreshold = 5f;
    public float timeInterval = 0.5f;

    [Header("Display")]
    public bool logToConsole = true;
    public bool showVisualization = true;

    private Vector3 lastPosition;
    private Quaternion lastRotation;
    private float nextUpdateTime = 0f;
    private List<string> lastDetection = new List<string>();

    public enum UpdateMode
    {
        OnMovement,
        TimedInterval,
        ManualOnly,
        EveryFrame // For debugging
    }

    void Start()
    {
        lastPosition = transform.position;
        lastRotation = transform.rotation;

        // Initial detection
        PerformDetection();
    }

    void Update()
    {
        switch (updateMode)
        {
            case UpdateMode.OnMovement:
                CheckMovementAndUpdate();
                break;

            case UpdateMode.TimedInterval:
                if (Time.time >= nextUpdateTime)
                {
                    PerformDetection();
                    nextUpdateTime = Time.time + timeInterval;
                }
                break;

            case UpdateMode.EveryFrame:
                PerformDetection();
                break;

            case UpdateMode.ManualOnly:
                // Only update via button or method call
                break;
        }

        // Manual trigger with Space key
        if (Input.GetKeyDown(KeyCode.Space))
        {
            PerformDetection();
        }
    }

    void CheckMovementAndUpdate()
    {
        float positionChange = Vector3.Distance(transform.position, lastPosition);
        float rotationChange = Quaternion.Angle(transform.rotation, lastRotation);

        if (positionChange > positionThreshold || rotationChange > rotationThreshold)
        {
            PerformDetection();
            lastPosition = transform.position;
            lastRotation = transform.rotation;
        }
    }

    public void PerformDetection()
    {
        if (rgbCamera == null)
        {
            Debug.LogWarning("RGB Camera not assigned!");
            return;
        }

        // Clear previous detection
        lastDetection.Clear();

        // Perform multi-ray detection
        Dictionary<Collider, float> detectedObjects = new Dictionary<Collider, float>();

        if (rayCount == 1)
        {
            // Single center ray
            Ray ray = rgbCamera.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0));
            RaycastHit[] hits = Physics.RaycastAll(ray, maxDistance, raycastMask);
            AddHitsToDictionary(hits, detectedObjects);
        }
        else
        {
            // Multiple rays in a grid
            int gridSize = Mathf.CeilToInt(Mathf.Sqrt(rayCount));

            for (int x = 0; x < gridSize; x++)
            {
                for (int y = 0; y < gridSize; y++)
                {
                    if (x * gridSize + y >= rayCount) break;

                    float u = (x + 0.5f) / gridSize;
                    float v = (y + 0.5f) / gridSize;

                    Ray ray = rgbCamera.ViewportPointToRay(new Vector3(u, v, 0));
                    RaycastHit[] hits = Physics.RaycastAll(ray, maxDistance, raycastMask);
                    AddHitsToDictionary(hits, detectedObjects);

                    if (showVisualization && Time.frameCount % 10 == 0)
                    {
                        Debug.DrawRay(ray.origin, ray.direction * maxDistance,
                                     Color.magenta, 0.5f);
                    }
                }
            }
        }

        // Process and display results
        ProcessDetectedObjects(detectedObjects);
    }

    void AddHitsToDictionary(RaycastHit[] hits, Dictionary<Collider, float> dictionary)
    {
        foreach (RaycastHit hit in hits)
        {
            if (!dictionary.ContainsKey(hit.collider) ||
                hit.distance < dictionary[hit.collider])
            {
                dictionary[hit.collider] = hit.distance;
            }
        }
    }

    void ProcessDetectedObjects(Dictionary<Collider, float> detectedObjects)
    {
        if (detectedObjects.Count == 0)
        {
            if (logToConsole)
                Debug.Log("No objects detected");
            return;
        }

        // Clear console
        ClearConsole();

        // Log new detection
        if (logToConsole)
        {
            Debug.Log($"=== Detection ({Time.time:F1}s) ===");
            Debug.Log($"Objects: {detectedObjects.Count}");
        }

        int index = 1;
        foreach (var kvp in detectedObjects)
        {
            Collider col = kvp.Key;
            float distance = kvp.Value;
            string objectInfo = $"{index}. {col.name} ({distance:F1}m)";

            lastDetection.Add(objectInfo);

            if (logToConsole)
                Debug.Log(objectInfo);

            index++;
        }
    }

    // Public method for UI buttons
    public void ManualDetection()
    {
        PerformDetection();
    }

    // Get results
    public List<string> GetDetectionResults()
    {
        return new List<string>(lastDetection);
    }

    // Clear Unity console
    void ClearConsole()
    {
#if UNITY_EDITOR
        var logEntries = System.Type.GetType("UnityEditor.LogEntries, UnityEditor");
        var clearMethod = logEntries?.GetMethod("Clear", 
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public);
        clearMethod?.Invoke(null, null);
#endif
    }

    void OnDrawGizmos()
    {
        if (showVisualization && rgbCamera != null)
        {
            // Draw camera forward direction
            Gizmos.color = Color.cyan;
            Gizmos.DrawRay(rgbCamera.transform.position,
                          rgbCamera.transform.forward * 2f);

            // Draw FOV visualization
            Gizmos.color = Color.yellow;
            float fov = rgbCamera.fieldOfView;
            float aspect = rgbCamera.aspect;
            float distance = 5f;

            Vector3[] corners = GetFrustumCorners(distance);
            for (int i = 0; i < 4; i++)
            {
                Gizmos.DrawLine(rgbCamera.transform.position, corners[i]);
            }
        }
    }

    Vector3[] GetFrustumCorners(float distance)
    {
        Vector3[] corners = new Vector3[4];
        float halfHeight = distance * Mathf.Tan(rgbCamera.fieldOfView * 0.5f * Mathf.Deg2Rad);
        float halfWidth = halfHeight * rgbCamera.aspect;

        corners[0] = rgbCamera.transform.TransformPoint(new Vector3(-halfWidth, -halfHeight, distance));
        corners[1] = rgbCamera.transform.TransformPoint(new Vector3(halfWidth, -halfHeight, distance));
        corners[2] = rgbCamera.transform.TransformPoint(new Vector3(halfWidth, halfHeight, distance));
        corners[3] = rgbCamera.transform.TransformPoint(new Vector3(-halfWidth, halfHeight, distance));

        return corners;
    }
}