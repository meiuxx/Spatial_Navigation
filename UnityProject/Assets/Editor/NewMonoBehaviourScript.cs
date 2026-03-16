using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

public class ExportWaypoints : EditorWindow
{
    [MenuItem("Tools/Export Waypoints to JSON")]
    static void Export()
    {
        int layer = LayerMask.NameToLayer("Waypoint");
        if (layer == -1)
        {
            EditorUtility.DisplayDialog("Error", "Layer 'Waypoint' not found. Please create it first.", "OK");
            return;
        }

        GameObject[] allObjects = GameObject.FindObjectsByType<GameObject>(FindObjectsSortMode.None);
        List<GameObject> waypointObjects = new List<GameObject>();

        // Collect all GameObjects on the Waypoint layer
        foreach (GameObject obj in allObjects)
        {
            if (obj.layer == layer)
                waypointObjects.Add(obj);
        }

        if (waypointObjects.Count == 0)
        {
            EditorUtility.DisplayDialog("No Waypoints", "No GameObjects found on the 'Waypoint' layer.", "OK");
            return;
        }

        // Sort by name using natural numeric order (e.g., Waypoint1, Waypoint2, ..., Waypoint10)
        waypointObjects.Sort((a, b) => CompareNames(a.name, b.name));

        // Extract positions (x = X, y = Z) in sorted order
        List<Vector2> waypoints = new List<Vector2>();
        foreach (GameObject obj in waypointObjects)
        {
            waypoints.Add(new Vector2(obj.transform.position.x, obj.transform.position.z));
        }

        // Convert to JSON
        string json = JsonUtility.ToJson(new WaypointList(waypoints), true);
        string path = EditorUtility.SaveFilePanel("Save Waypoints JSON", Application.dataPath, "waypoints.json", "json");
        if (!string.IsNullOrEmpty(path))
        {
            File.WriteAllText(path, json);
            AssetDatabase.Refresh();
            EditorUtility.DisplayDialog("Export Complete", $"Exported {waypoints.Count} waypoints to:\n{path}", "OK");
        }
    }

    // Helper method to compare strings with numeric suffixes naturally
    private static int CompareNames(string nameA, string nameB)
    {
        // Extract numeric part from the end of the name
        int numA = ExtractNumber(nameA);
        int numB = ExtractNumber(nameB);
        return numA.CompareTo(numB);
    }

    private static int ExtractNumber(string name)
    {
        // Find the last continuous digits in the string
        Match match = Regex.Match(name, @"\d+$");
        if (match.Success)
            return int.Parse(match.Value);
        return 0; // fallback if no number
    }

    [System.Serializable]
    private class WaypointList
    {
        public List<Vector2> waypoints;
        public WaypointList(List<Vector2> points) => waypoints = points;
    }
}