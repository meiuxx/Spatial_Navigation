using UnityEngine;

public class AgentVision : MonoBehaviour
{
    public Camera rgbCamera;
    public int width = 224;
    public int height = 224;

    private RenderTexture renderTexture;
    private Texture2D frameTexture;

    void Awake()
    {
        renderTexture = new RenderTexture(width, height, 24);
        frameTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
        rgbCamera.targetTexture = renderTexture;
    }

    public byte[] CaptureRGB()
    {
        RenderTexture.active = renderTexture;
        rgbCamera.Render();

        frameTexture.ReadPixels(
            new Rect(0, 0, width, height), 0, 0
        );
        frameTexture.Apply();

        RenderTexture.active = null;


        return frameTexture.EncodeToPNG(); // or JPG
    }
}
