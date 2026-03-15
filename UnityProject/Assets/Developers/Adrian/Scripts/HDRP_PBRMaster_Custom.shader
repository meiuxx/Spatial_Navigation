Shader "Custom/HDRP_PBRMaster_FallbackOpaque"
{
    Properties
    {
        _BaseMap ("Base Color", 2D) = "white" {}
        _NormalMap ("Normal Map", 2D) = "bump" {}
        _ORMMap ("ORM (R=AO G=Rough B=Metal)", 2D) = "white" {}

        _UVScale ("UV Scale", Vector) = (1,1,0,0)
        _UVOffset ("UV Offset", Vector) = (0,0,0,0)
        _UVMatrix ("UV Rotation Matrix", Vector) = (1,0,0,1)

        _NormalStrength ("Normal Strength", Range(0,2)) = 1

        _ColorIntensity ("Color Intensity", Float) = 1
        _ColorPower ("Color Power", Float) = 1
        _Desaturate ("Desaturate", Range(0,1)) = 0
        _Tint ("Tint", Color) = (1,1,1,1)

        _MetallicScale ("Metallic Scale", Float) = 1
        _MetallicPower ("Metallic Power", Float) = 1
        _RoughnessScale ("Roughness Scale", Float) = 1
        _RoughnessPower ("Roughness Power", Float) = 1

        _EmissiveA ("Emissive Color A", Color) = (0,0,0,1)
        _EmissiveB ("Emissive Color B", Color) = (0,0,0,1)
        _EmissiveBlend ("Emissive Blend", Range(0,1)) = 0

        _LightDirection ("Light Direction", Vector) = (0.3,0.8,0.4,0)
        _LightColor ("Light Color", Color) = (1,1,1,1)
        _AmbientColor ("Ambient Color", Color) = (0.25,0.25,0.25,1)
    }

    SubShader
    {
        Tags
        {
            "Queue"="Geometry"
            "RenderType"="Opaque"
        }

        Pass
        {
            Name "Forward"
            Tags { "LightMode"="ForwardBase" }

            Cull Back
            ZWrite On
            ZTest LEqual
            Blend Off

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            sampler2D _BaseMap;
            sampler2D _NormalMap;
            sampler2D _ORMMap;

            float4 _UVScale;
            float4 _UVOffset;
            float4 _UVMatrix;

            float _NormalStrength;

            float _ColorIntensity;
            float _ColorPower;
            float _Desaturate;
            float4 _Tint;

            float _MetallicScale;
            float _MetallicPower;
            float _RoughnessScale;
            float _RoughnessPower;

            float4 _EmissiveA;
            float4 _EmissiveB;
            float _EmissiveBlend;

            float4 _LightDirection;
            float4 _LightColor;
            float4 _AmbientColor;

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 positionWS : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
                float3 tangentWS : TEXCOORD3;
                float3 bitangentWS : TEXCOORD4;
            };

            float2 TransformCustomUV(float2 uv)
            {
                uv -= 0.5;

                float2x2 m = float2x2(
                    _UVMatrix.x, _UVMatrix.y,
                    _UVMatrix.z, _UVMatrix.w
                );

                uv = mul(m, uv);
                uv += 0.5;
                uv = uv * _UVScale.xy + _UVOffset.xy;

                return uv;
            }

            float3 UnpackNormalSimple(float4 c)
            {
                float3 n;
                n.xy = c.xy * 2.0 - 1.0;
                n.xy *= _NormalStrength;
                n.z = sqrt(saturate(1.0 - dot(n.xy, n.xy)));
                return normalize(n);
            }

            float D_GGX(float NoH, float roughness)
            {
                float a = max(roughness * roughness, 0.001);
                float a2 = a * a;
                float denom = (NoH * NoH) * (a2 - 1.0) + 1.0;
                return a2 / max(UNITY_PI * denom * denom, 0.0001);
            }

            float G_Smith(float NoV, float NoL, float roughness)
            {
                float r = roughness + 1.0;
                float k = (r * r) / 8.0;

                float gv = NoV / lerp(NoV, 1.0, k);
                float gl = NoL / lerp(NoL, 1.0, k);
                return gv * gl;
            }

            float3 F_Schlick(float VoH, float3 F0)
            {
                float f = pow(1.0 - VoH, 5.0);
                return F0 + (1.0 - F0) * f;
            }

            v2f vert(appdata v)
            {
                v2f o;

                o.pos = UnityObjectToClipPos(v.vertex);
                o.positionWS = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.normalWS = UnityObjectToWorldNormal(v.normal);
                o.tangentWS = UnityObjectToWorldDir(v.tangent.xyz);
                o.bitangentWS = cross(o.normalWS, o.tangentWS) * v.tangent.w;
                o.uv = TransformCustomUV(v.uv);

                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float2 uv = i.uv;

                float3 baseColor = tex2D(_BaseMap, uv).rgb;
                float4 normalTex = tex2D(_NormalMap, uv);
                float3 orm = tex2D(_ORMMap, uv).rgb;

                baseColor = pow(max(baseColor * _ColorIntensity, 0.0001), _ColorPower);

                float luma = dot(baseColor, float3(0.2126, 0.7152, 0.0722));
                baseColor = lerp(baseColor, luma.xxx, saturate(_Desaturate));
                baseColor *= _Tint.rgb;

                float ao = saturate(orm.r);
                float roughness = saturate(pow(max(orm.g * _RoughnessScale, 0.0001), _RoughnessPower));
                float metallic = saturate(pow(max(orm.b * _MetallicScale, 0.0001), _MetallicPower));

                float3 emission = lerp(_EmissiveA.rgb, _EmissiveB.rgb, saturate(_EmissiveBlend));

                float3 normalTS = UnpackNormalSimple(normalTex);

                float3x3 tbn = float3x3(
                    normalize(i.tangentWS),
                    normalize(i.bitangentWS),
                    normalize(i.normalWS)
                );

                float3 N = normalize(mul(normalTS, tbn));
                float3 V = normalize(_WorldSpaceCameraPos.xyz - i.positionWS);
                float3 L = normalize(_LightDirection.xyz);
                float3 H = normalize(L + V);

                float NoL = saturate(dot(N, L));
                float NoV = saturate(dot(N, V));
                float NoH = saturate(dot(N, H));
                float VoH = saturate(dot(V, H));

                float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);

                float D = D_GGX(NoH, roughness);
                float G = G_Smith(max(NoV, 0.0001), max(NoL, 0.0001), roughness);
                float3 F = F_Schlick(VoH, F0);

                float3 specular = (D * G * F) / max(4.0 * max(NoV, 0.0001) * max(NoL, 0.0001), 0.0001);

                float3 kS = F;
                float3 kD = (1.0 - kS) * (1.0 - metallic);
                float3 diffuse = kD * baseColor / UNITY_PI;

                float3 direct = (diffuse + specular) * _LightColor.rgb * NoL;
                float3 ambient = _AmbientColor.rgb * baseColor * ao;

                float3 finalColor = direct + ambient + emission;

                return float4(finalColor, 1.0);
            }
            ENDHLSL
        }
    }
}