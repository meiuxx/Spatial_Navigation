using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Editor tool to find and delete unused textures from the project.
/// Place this file in any folder named "Editor" inside your Assets directory.
/// Access via: Tools > Clean Up > Delete Unused Textures
/// </summary>
public class UnusedTextureCleaner : EditorWindow
{
    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    private List<string> _unusedTexturePaths = new();
    private Vector2 _scrollPos;
    private bool _hasScanned = false;
    private bool _isBusy = false;

    // Texture types Unity recognises
    private static readonly string[] TextureExtensions =
    {
        ".png", ".jpg", ".jpeg", ".tga", ".psd",
        ".tiff", ".tif", ".bmp", ".gif", ".exr",
        ".hdr", ".iff", ".pict", ".dds", ".ktx"
    };

    // -----------------------------------------------------------------------
    // Menu entry
    // -----------------------------------------------------------------------
    [MenuItem("Tools/Clean Up/Delete Unused Textures")]
    public static void ShowWindow()
    {
        var window = GetWindow<UnusedTextureCleaner>("Unused Texture Cleaner");
        window.minSize = new Vector2(540, 420);
        window.Show();
    }

    // -----------------------------------------------------------------------
    // GUI
    // -----------------------------------------------------------------------
    private void OnGUI()
    {
        GUILayout.Label("Unused Texture Cleaner", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox(
            "Scans every scene, prefab, material, ScriptableObject and other asset " +
            "in the project to find textures that are never referenced.\n\n" +
            "⚠  Always back up your project (or use source control) before deleting assets.",
            MessageType.Warning);

        EditorGUILayout.Space(6);

        // --- Scan button ---
        GUI.enabled = !_isBusy;
        if (GUILayout.Button("Scan for Unused Textures", GUILayout.Height(32)))
            RunScan();

        EditorGUILayout.Space(4);

        // --- Results ---
        if (_hasScanned)
        {
            if (_unusedTexturePaths.Count == 0)
            {
                EditorGUILayout.HelpBox("No unused textures found. Your project is clean!", MessageType.Info);
            }
            else
            {
                EditorGUILayout.LabelField($"Found {_unusedTexturePaths.Count} unused texture(s):",
                    EditorStyles.boldLabel);

                _scrollPos = EditorGUILayout.BeginScrollView(_scrollPos,
                    GUILayout.ExpandHeight(true));

                foreach (var path in _unusedTexturePaths)
                {
                    EditorGUILayout.BeginHorizontal();

                    // Thumbnail
                    var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
                    if (tex != null)
                        GUILayout.Label(tex, GUILayout.Width(40), GUILayout.Height(40));
                    else
                        GUILayout.Space(44);

                    // Path label — clicking pings it in the Project window
                    if (GUILayout.Button(path, EditorStyles.linkLabel,
                            GUILayout.ExpandWidth(true)))
                        EditorGUIUtility.PingObject(
                            AssetDatabase.LoadAssetAtPath<Object>(path));

                    EditorGUILayout.EndHorizontal();
                }

                EditorGUILayout.EndScrollView();

                EditorGUILayout.Space(4);

                // --- Delete button ---
                GUI.backgroundColor = new Color(1f, 0.35f, 0.35f);
                if (GUILayout.Button(
                        $"Delete All {_unusedTexturePaths.Count} Unused Texture(s)",
                        GUILayout.Height(32)))
                {
                    if (ConfirmDeletion())
                        DeleteUnusedTextures();
                }
                GUI.backgroundColor = Color.white;
            }
        }

        GUI.enabled = true;
    }

    // -----------------------------------------------------------------------
    // Core logic
    // -----------------------------------------------------------------------

    /// <summary>
    /// Collects every texture asset in the project, then removes those that
    /// appear in at least one dependency of any other asset or scene.
    /// </summary>
    private void RunScan()
    {
        _isBusy = true;
        _unusedTexturePaths.Clear();
        _hasScanned = false;

        try
        {
            // 1. Gather ALL texture GUIDs in the project
            var allTextureGuids = new HashSet<string>(
                AssetDatabase.FindAssets("t:Texture", new[] { "Assets" }));

            if (allTextureGuids.Count == 0)
            {
                Debug.Log("[UnusedTextureCleaner] No textures found in project.");
                _hasScanned = true;
                return;
            }

            // 2. Build the set of all REFERENCED asset GUIDs via dependency graph.
            //    We look at every non-texture asset (scenes, prefabs, materials, etc.)
            //    plus every texture itself (a texture can reference another texture,
            //    e.g. a render texture used as source).
            var referencedGuids = new HashSet<string>();

            var allAssetGuids = AssetDatabase.FindAssets("", new[] { "Assets" });
            int total = allAssetGuids.Length;

            for (int i = 0; i < total; i++)
            {
                string guid = allAssetGuids[i];
                string path = AssetDatabase.GUIDToAssetPath(guid);

                // Skip if this IS a texture — we don't want a texture to keep
                // itself alive just because it's in the dependency list.
                if (IsTextureFile(path) && allTextureGuids.Contains(guid))
                    continue;

                EditorUtility.DisplayProgressBar(
                    "Scanning dependencies…",
                    Path.GetFileName(path),
                    (float)i / total);

                // GetDependencies returns paths, not GUIDs
                string[] deps = AssetDatabase.GetDependencies(path, recursive: true);
                foreach (string dep in deps)
                {
                    string depGuid = AssetDatabase.AssetPathToGUID(dep);
                    if (!string.IsNullOrEmpty(depGuid))
                        referencedGuids.Add(depGuid);
                }
            }

            // 3. Any texture GUID not in referencedGuids is unused
            foreach (string guid in allTextureGuids)
            {
                if (!referencedGuids.Contains(guid))
                {
                    string path = AssetDatabase.GUIDToAssetPath(guid);
                    if (!string.IsNullOrEmpty(path))
                        _unusedTexturePaths.Add(path);
                }
            }

            _unusedTexturePaths.Sort();
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            _hasScanned = true;
            _isBusy = false;
            Repaint();
        }

        Debug.Log($"[UnusedTextureCleaner] Scan complete. " +
                  $"{_unusedTexturePaths.Count} unused texture(s) found.");
    }

    /// <summary>
    /// Permanently deletes every path in <see cref="_unusedTexturePaths"/>.
    /// </summary>
    private void DeleteUnusedTextures()
    {
        _isBusy = true;
        int deleted = 0;
        var failed = new List<string>();

        try
        {
            for (int i = 0; i < _unusedTexturePaths.Count; i++)
            {
                string path = _unusedTexturePaths[i];

                EditorUtility.DisplayProgressBar(
                    "Deleting unused textures…",
                    Path.GetFileName(path),
                    (float)i / _unusedTexturePaths.Count);

                if (AssetDatabase.DeleteAsset(path))
                    deleted++;
                else
                    failed.Add(path);
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            _isBusy = false;
        }

        _unusedTexturePaths.Clear();
        _hasScanned = false;

        if (failed.Count > 0)
        {
            Debug.LogWarning(
                $"[UnusedTextureCleaner] Could not delete {failed.Count} asset(s):\n" +
                string.Join("\n", failed));
        }

        Debug.Log($"[UnusedTextureCleaner] Done. {deleted} texture(s) deleted.");
        EditorUtility.DisplayDialog("Done",
            $"Deleted {deleted} unused texture(s)." +
            (failed.Count > 0 ? $"\n\n{failed.Count} could not be deleted (see Console)." : ""),
            "OK");

        Repaint();
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static bool IsTextureFile(string path)
    {
        if (string.IsNullOrEmpty(path)) return false;
        string ext = Path.GetExtension(path).ToLowerInvariant();
        return TextureExtensions.Contains(ext);
    }

    private static bool ConfirmDeletion()
    {
        return EditorUtility.DisplayDialog(
            "Delete Unused Textures",
            "This will permanently delete the listed textures from disk.\n\n" +
            "Make sure you have committed or backed up your project first.\n\n" +
            "Continue?",
            "Yes, Delete",
            "Cancel");
    }
}