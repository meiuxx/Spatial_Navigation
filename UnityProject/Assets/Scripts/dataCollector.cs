using UnityEngine;
using System.IO;
using System.Text;

public class TrainingDataCapture : MonoBehaviour
{
    [Header("Settings")]
    public KeyCode captureKey = KeyCode.P;          // Key to press for capturing
    public int captureWidth = 640;                   // Width of saved image
    public int captureHeight = 480;                   // Height of saved image
    public string dataFolder = "TrainingData";       // Subfolder name inside persistentDataPath

    private Camera cam;
    private string imageFolder;
    private string csvPath;
    private int imageCounter = 0;

    void Start()
    {
        cam = GetComponent<Camera>();
        if (cam == null)
        {
            Debug.LogError("TrainingDataCapture: No Camera component found on this GameObject.");
            enabled = false;
            return;
        }

        // Create folders
        string basePath = Path.Combine(Application.persistentDataPath, dataFolder);
        imageFolder = Path.Combine(basePath, "Images");
        Directory.CreateDirectory(imageFolder);
        csvPath = Path.Combine(basePath, "metadata.csv");

        // Write CSV header if file doesn't exist
        if (!File.Exists(csvPath))
        {
            using (StreamWriter sw = new StreamWriter(csvPath, false, Encoding.UTF8))
            {
                sw.WriteLine("filename,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,timestamp");
            }
        }

        // Count existing images to avoid overwriting
        string[] files = Directory.GetFiles(imageFolder, "*.png");
        imageCounter = files.Length;

        Debug.Log($"Training data will be saved to: {basePath}");
    }

    void Update()
    {
        if (Input.GetKeyDown(captureKey))
        {
            Capture();
        }
    }

    void Capture()
    {
        // Create a render texture and temporarily set the camera to render into it
        RenderTexture rt = new RenderTexture(captureWidth, captureHeight, 24);
        RenderTexture originalRT = cam.targetTexture;
        cam.targetTexture = rt;

        // Render the camera's view
        Texture2D screenShot = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        cam.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        screenShot.Apply();

        // Restore original camera settings
        cam.targetTexture = originalRT;
        RenderTexture.active = null;
        Destroy(rt);

        // Encode to PNG
        byte[] bytes = screenShot.EncodeToPNG();
        Destroy(screenShot);

        // Generate filename with zero-padded counter
        string filename = $"img_{imageCounter:D6}.png";
        string filePath = Path.Combine(imageFolder, filename);

        // Save image
        File.WriteAllBytes(filePath, bytes);
        Debug.Log($"Saved: {filePath}");

        // Get robot pose
        Vector3 pos = transform.position;
        Quaternion rot = transform.rotation;
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");

        // Append to CSV
        using (StreamWriter sw = new StreamWriter(csvPath, true, Encoding.UTF8))
        {
            sw.WriteLine($"{filename},{pos.x},{pos.y},{pos.z},{rot.x},{rot.y},{rot.z},{rot.w},{timestamp}");
        }

        imageCounter++;
    }
}