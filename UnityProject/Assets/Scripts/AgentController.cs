using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class RobotKeyController : MonoBehaviour
{
    [Header("Movement Settings")]
    public float moveSpeed = 5.0f;
    public float turnSpeed = 100.0f;

    private Rigidbody rb;
    private float moveInput;
    private float turnInput;

    void Start()
    {
        rb = GetComponent<Rigidbody>();

        // Essential for robot simulation: 
        // Prevents the robot from tipping over but allows linear movement
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        // High drag prevents the robot from sliding forever
        rb.linearDamping = 2.0f;
        rb.angularDamping = 2.0f;
    }

    void Update()
    {
        // Capture WASD or Arrow Key inputs
        moveInput = Input.GetAxis("Vertical");   // W/S
        turnInput = Input.GetAxis("Horizontal"); // A/D
    }

    void FixedUpdate()
    {
        MoveRobot();
        TurnRobot();
    }

    void MoveRobot()
    {
        // Move forward/backward relative to the robot's orientation
        Vector3 movement = transform.forward * moveInput * moveSpeed;
        rb.AddForce(movement, ForceMode.Acceleration);
    }

    void TurnRobot()
    {
        // Rotate around the Y axis
        float rotation = turnInput * turnSpeed * Time.fixedDeltaTime;
        Quaternion turnRotation = Quaternion.Euler(0f, rotation, 0f);
        rb.MoveRotation(rb.rotation * turnRotation);
    }
}