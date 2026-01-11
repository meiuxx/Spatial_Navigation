using UnityEngine;

[System.Serializable]
public class AgentObservation: MonoBehaviour
{
    public Texture2D rgb;
    public Vector3[] rayDirections;
    public float[] rayDistances;

    public Vector3 agentPosition;
    public Vector3 agentRotation;
}
