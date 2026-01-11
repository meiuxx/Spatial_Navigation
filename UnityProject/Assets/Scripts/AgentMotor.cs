using UnityEngine;

public class AgentMotor : MonoBehaviour
{
    public float moveSpeed = 2f;   // meters/sec
    public float turnSpeed = 90f;  // degrees/sec

    private bool isMoving = false;
    private bool isTurning = false;

    private float targetDistance;
    private float currentDistance;
    private float moveDirection = 1f; // 1 = forward, -1 = backward

    private float targetAngle;
    private float currentAngle;
    private float turnDirection = 1f; // 1 = right, -1 = left

    void Update()
    {
        // --- MOVE ---
        if (isMoving)
        {
            float step = moveSpeed * Time.deltaTime * moveDirection;
            transform.Translate(Vector3.forward * step);
            currentDistance += Mathf.Abs(step);

            if (currentDistance >= targetDistance)
            {
                isMoving = false;
                currentDistance = 0f;
                moveDirection = 1f; // reset
            }
        }

        // --- TURN ---
        if (isTurning)
        {
            float step = turnSpeed * Time.deltaTime * turnDirection;
            transform.Rotate(Vector3.up * step);
            currentAngle += Mathf.Abs(step);

            if (currentAngle >= Mathf.Abs(targetAngle))
            {
                isTurning = false;
                currentAngle = 0f;
                turnDirection = 1f; // reset
            }
        }
    }

    // ===== COMMAND INTERFACE =====

    public void MoveForward(float distance = 1f)
    {
        if (isMoving) return;
        isMoving = true;
        targetDistance = distance;
        moveDirection = 1f;
        currentDistance = 0f;
    }

    public void MoveBackward(float distance = 1f)
    {
        if (isMoving) return;
        isMoving = true;
        targetDistance = distance;
        moveDirection = -1f;
        currentDistance = 0f;
    }

    public void TurnLeft(float angle = 45f)
    {
        if (isTurning) return;
        isTurning = true;
        targetAngle = angle;
        turnDirection = -1f;
        currentAngle = 0f;
    }

    public void TurnRight(float angle = 45f)
    {
        if (isTurning) return;
        isTurning = true;
        targetAngle = angle;
        turnDirection = 1f;
        currentAngle = 0f;
    }

    public void StopMoving()
    {
        isMoving = false;
        currentDistance = 0f;
        moveDirection = 1f;
    }

    public void StopTurning()
    {
        isTurning = false;
        currentAngle = 0f;
        turnDirection = 1f;
    }
}
