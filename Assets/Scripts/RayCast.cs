using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class RGBRaycast : MonoBehaviour
{
    public Camera rgbCamera;

    [Header("Detection Settings")]
    public float maxDistance = 100f;
    public LayerMask raycastMask = -1;
    public int rayCount = 9;

    [Header("Debug")]
    public bool debugMode = true;

    [System.Serializable]
    public class RayObservation
    {
        public string objectName;
        public float distance;
    }

    private List<RayObservation> lastDetection = new List<RayObservation>();

    void Start()
    {
        if (rgbCamera == null)
        {
            rgbCamera = GetComponent<Camera>();
            if (rgbCamera == null)
            {
                rgbCamera = Camera.main;
            }
        }

        if (debugMode)
        {
            Debug.Log($"RGBRaycast initialized with camera: {rgbCamera?.gameObject.name}");
            Debug.Log($"Raycast mask value: {raycastMask.value}");
        }
    }

    public void PerformDetection()
    {
        if (rgbCamera == null)
        {
            Debug.LogError("RGB Camera not assigned!");
            return;
        }

        // Debug info
        if (debugMode)
        {
            Debug.Log($"=== Performing Raycast Detection ===");
            Debug.Log($"Camera: {rgbCamera.name}");
            Debug.Log($"Position: {rgbCamera.transform.position}");
            Debug.Log($"Forward: {rgbCamera.transform.forward}");
            Debug.Log($"Max distance: {maxDistance}");
            Debug.Log($"Ray count: {rayCount}");
        }

        // Clear previous detection
        lastDetection.Clear();

        // Perform multi-ray detection
        Dictionary<Collider, RayObservation> detectedObjects = new Dictionary<Collider, RayObservation>();

        if (rayCount == 1)
        {
            // Single center ray
            Ray ray = rgbCamera.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0));
            if (debugMode)
            {
                Debug.Log($"Center ray: origin={ray.origin}, direction={ray.direction}");
                Debug.DrawRay(ray.origin, ray.direction * maxDistance, Color.red, 2f);
            }

            RaycastHit[] hits = Physics.RaycastAll(ray, maxDistance, raycastMask);

            if (debugMode) Debug.Log($"Center ray hits: {hits.Length}");

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

                    if (debugMode)
                    {
                        Debug.DrawRay(ray.origin, ray.direction * maxDistance,
                                     new Color(u, v, 0.5f, 0.5f), 2f);
                    }
                }
            }
        }

        // Convert to list
        lastDetection = detectedObjects.Values.ToList();

        // Sort by distance
        lastDetection.Sort((a, b) => a.distance.CompareTo(b.distance));

        // Debug output
        if (debugMode)
        {
            if (lastDetection.Count == 0)
            {
                Debug.LogWarning("No objects detected by raycast!");

                // Debug: Draw all colliders in scene
                Collider[] allColliders = FindObjectsOfType<Collider>();
                Debug.Log($"Total colliders in scene: {allColliders.Length}");
                foreach (Collider col in allColliders)
                {
                    Debug.Log($"  - {col.gameObject.name} at {col.transform.position}");
                }
            }
            else
            {
                Debug.Log($"Detected {lastDetection.Count} objects:");
                foreach (var obs in lastDetection)
                {
                    Debug.Log($"  • {obs.objectName} @ {obs.distance:F2}m");
                }
            }
        }
    }

    void AddHitsToDictionary(RaycastHit[] hits, Dictionary<Collider, RayObservation> dictionary)
    {
        foreach (RaycastHit hit in hits)
        {
            if (!dictionary.ContainsKey(hit.collider) ||
                hit.distance < dictionary[hit.collider].distance)
            {
                dictionary[hit.collider] = new RayObservation
                {
                    objectName = hit.collider.gameObject.name,
                    distance = hit.distance
                };

                if (debugMode)
                {
                    Debug.Log($"Hit: {hit.collider.name} at {hit.distance:F2}m, point: {hit.point}");
                    Debug.DrawLine(rgbCamera.transform.position, hit.point, Color.green, 2f);
                }
            }
        }
    }

    public List<RayObservation> GetRayObservations()
    {
        return new List<RayObservation>(lastDetection);
    }

    public int GetDetectionCount()
    {
        return lastDetection.Count;
    }

    void OnDrawGizmosSelected()
    {
        if (rgbCamera != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(rgbCamera.transform.position, 0.1f);

            // Draw camera frustum
            float distance = 5f;
            float halfHeight = distance * Mathf.Tan(rgbCamera.fieldOfView * 0.5f * Mathf.Deg2Rad);
            float halfWidth = halfHeight * rgbCamera.aspect;

            Vector3 center = rgbCamera.transform.position + rgbCamera.transform.forward * distance;
            Vector3 topLeft = center + (-rgbCamera.transform.right * halfWidth) + (rgbCamera.transform.up * halfHeight);
            Vector3 topRight = center + (rgbCamera.transform.right * halfWidth) + (rgbCamera.transform.up * halfHeight);
            Vector3 bottomLeft = center + (-rgbCamera.transform.right * halfWidth) + (-rgbCamera.transform.up * halfHeight);
            Vector3 bottomRight = center + (rgbCamera.transform.right * halfWidth) + (-rgbCamera.transform.up * halfHeight);

            Gizmos.DrawLine(rgbCamera.transform.position, topLeft);
            Gizmos.DrawLine(rgbCamera.transform.position, topRight);
            Gizmos.DrawLine(rgbCamera.transform.position, bottomLeft);
            Gizmos.DrawLine(rgbCamera.transform.position, bottomRight);
        }
    }
}