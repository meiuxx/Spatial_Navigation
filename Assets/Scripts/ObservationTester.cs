using UnityEngine;

public class ObservationTester : MonoBehaviour
{
    public ObservationCollector collector;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.T))
        {
            if (collector != null)
            {
                collector.CollectAndSend();
                Debug.Log("[Test] Observation sent!");
            }
            else
            {
                Debug.LogWarning("[Test] ObservationCollector not assigned!");
            }
        }
    }

}
