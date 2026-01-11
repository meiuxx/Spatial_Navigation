using UnityEngine;

public class AgentController : MonoBehaviour
{
    public float moveSpeed = 2f;
    public float turnSpeed = 90f;

    void Update()
    {
        float move = Input.GetAxis("Vertical");
        float turn = Input.GetAxis("Horizontal");

        transform.Translate(Vector3.forward * move * moveSpeed * Time.deltaTime);
        transform.Rotate(Vector3.up * turn * turnSpeed * Time.deltaTime);
    }
}