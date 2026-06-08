using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif
#if HDRP_PRESENT
using UnityEngine.Rendering.HighDefinition;
#endif

/// <summary>
/// SignLightFixture — A focused Spot Light for illuminating a sign.
///
/// TARGET PIPELINE: High Definition Render Pipeline (HDRP)
///   • Spot Light uses HDAdditionalLightData with Lumen intensity units,
///     area shadows, and contact shadows — all standard HDRP settings.
///
/// QUICK SETUP:
///   1. Attach this component to an empty GameObject.
///   2. Position the GameObject above your sign.
///   3. Hit Play, or click away — [ExecuteAlways] rebuilds in Edit mode too.
///
/// The light automatically aims at the sign face if <signTarget> is assigned.
/// Without a target the light aims straight down (-Y in local space).
/// </summary>
[ExecuteAlways]
public class SignLightFixture : MonoBehaviour
{
    // ─── Spotlight ────────────────────────────────────────────────────────────
    [Header("Spot Light")]
    [Tooltip("Optional: the sign GameObject or transform the light should aim at.")]
    public Transform signTarget = null;
    public Color lightColor = new Color(1f, 0.97f, 0.88f);   // warm white

    [Tooltip("HDRP: light intensity in Lumens. A typical 50 W halogen sign light is ~600–800 lm.")]
    [Range(100f, 5000f)]
    public float lightIntensityLumens = 700f;

    [Tooltip("HDRP: maximum range (metres) before the light is culled entirely.")]
    [Range(0.5f, 20f)]
    public float lightRange = 3.0f;
    [Range(5f, 60f)]
    public float spotAngle = 30f;
    [Range(1f, 20f)]
    public float innerSpotPercent = 60f;   // soft inner cone as % of spotAngle

    [Tooltip("HDRP: enables per-pixel contact shadows on the spot light (small cost, big quality gain).")]
    public bool useContactShadows = true;
    [Tooltip("HDRP: enables area (soft) shadows. Requires shadow maps to be enabled in HDRP asset.")]
    public bool useAreaShadows = true;
    [Tooltip("HDRP: colour temperature in Kelvin. 3200 K = warm halogen, 4000 K = neutral white, 6500 K = daylight.")]
    [Range(1500f, 10000f)]
    public float colourTemperature = 3200f;
    [Tooltip("HDRP: enable physical colour temperature filtering (overrides the Color tint above when enabled).")]
    public bool useColourTemperature = false;

    [Header("Flicker (optional)")]
    [Tooltip("Simulate an old-style flicker — leave at 0 to disable.")]
    [Range(0f, 1f)]
    public float flickerStrength = 0f;
    [Range(0f, 30f)]
    public float flickerFrequency = 8f;

    // ─── Private references ───────────────────────────────────────────────────
    private Light _spotLight;
    private float _baseIntensityLumens;
    private float _flickerTimer;

    // ─────────────────────────────────────────────────────────────────────────
    //  Unity callbacks
    // ─────────────────────────────────────────────────────────────────────────

    private void OnEnable() => BuildFixture();
    private void OnValidate() => BuildFixture();

    private void Update()
    {
        AimAtTarget();
        HandleFlicker();
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Destroy and rebuild the spot light from scratch.</summary>
    public void BuildFixture()
    {
        DestroyChildren();
        BuildSpotLight();
        _baseIntensityLumens = lightIntensityLumens;
    }

    /// <summary>Toggle the light on/off without rebuilding.</summary>
    public void SetLightEnabled(bool on)
    {
        if (_spotLight != null) _spotLight.enabled = on;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Build helpers
    // ─────────────────────────────────────────────────────────────────────────

    private void BuildSpotLight()
    {
        var lightGO = new GameObject("Sign_SpotLight");
        lightGO.transform.SetParent(transform, false);
        lightGO.transform.localPosition = Vector3.zero;
        lightGO.transform.localRotation = Quaternion.identity;

        _spotLight = lightGO.AddComponent<Light>();
        _spotLight.type = LightType.Spot;
        _spotLight.color = lightColor;
        _spotLight.range = lightRange;
        _spotLight.spotAngle = spotAngle;
        _spotLight.innerSpotAngle = spotAngle * (innerSpotPercent / 100f);
        _spotLight.shadows = useAreaShadows ? LightShadows.Soft : LightShadows.None;
        _spotLight.shadowStrength = 0.85f;
        _spotLight.intensity = 1f;
        _spotLight.renderMode = LightRenderMode.ForcePixel;

#if HDRP_PRESENT
        var hdLight = lightGO.GetComponent<HDAdditionalLightData>();
        if (hdLight == null) hdLight = lightGO.AddComponent<HDAdditionalLightData>();

        hdLight.lightUnit  = LightUnit.Lumen;
        hdLight.intensity  = lightIntensityLumens;

        hdLight.enableColorTemperature = useColourTemperature;
        if (useColourTemperature)
            hdLight.colorTemperature = colourTemperature;

        hdLight.EnableShadows(useAreaShadows);
        hdLight.SetShadowResolution(HDShadowResolution.FromQualitySettings);

        hdLight.useContactShadow.useOverride = useContactShadows;
        hdLight.useContactShadow.@override   = useContactShadows;

        hdLight.affectDiffuse  = true;
        hdLight.affectSpecular = true;

        hdLight.volumetricDimmer = 0.5f;
        hdLight.lightDimmer      = 1.0f;
#endif
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Runtime helpers
    // ─────────────────────────────────────────────────────────────────────────

    private void AimAtTarget()
    {
        if (_spotLight == null) return;

        if (signTarget != null)
            transform.LookAt(signTarget.position, Vector3.up);
    }

    private void HandleFlicker()
    {
        if (_spotLight == null || flickerStrength <= 0f) return;

        _flickerTimer += Time.deltaTime * flickerFrequency;
        float noise = Mathf.PerlinNoise(_flickerTimer, 0f) * 2f - 1f;
        float flickeredLm = _baseIntensityLumens + noise * flickerStrength * _baseIntensityLumens;

#if HDRP_PRESENT
        var hdLight = _spotLight.GetComponent<HDAdditionalLightData>();
        if (hdLight != null)
        {
            hdLight.intensity = Mathf.Max(0f, flickeredLm);
            return;
        }
#endif
        _spotLight.intensity = Mathf.Max(0f, flickeredLm / 700f);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Utility
    // ─────────────────────────────────────────────────────────────────────────

    private void DestroyChildren()
    {
        var children = new System.Collections.Generic.List<GameObject>();
        foreach (Transform child in transform)
            children.Add(child.gameObject);

        foreach (var child in children)
        {
            if (Application.isPlaying) Destroy(child);
            else DestroyImmediate(child);
        }

        _spotLight = null;
    }
}