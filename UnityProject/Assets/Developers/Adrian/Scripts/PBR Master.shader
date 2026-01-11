Shader "Universal Render Pipeline/_PBR Master"
{
    Properties
    {
        _BaseMap ("Base Color", 2D) = "white" {}
        _NormalMap ("Normal Map (Tangent Space)", 2D) = "bump" {}
        _ORMMap ("ORM (R=AO G=Rough B=Metal)", 2D) = "white" {}

        _UVScale ("UV Scale", Vector) = (1,1,0,0)
        _UVOffset ("UV Offset", Vector) = (0,0,0,0)
        _UVMatrix ("UV Rotation Matrix", Vector) = (1,0,0,1)

        _NormalStrength ("Normal Strength", Range(-1,2)) = 1

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
    }

    SubShader
    {
        Tags
        {
            "RenderType"="Opaque"
            "RenderPipeline"="UniversalPipeline"
            "UniversalMaterialType"="Lit"
        }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode"="UniversalForward" }

            ZWrite On
            Cull Back

            HLSLPROGRAM
            #pragma target 2.0
            #pragma vertex vert
            #pragma fragment frag

            // Match URP/Lit forward keywords more closely
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT _SHADOWS_SOFT_LOW _SHADOWS_SOFT_MEDIUM _SHADOWS_SOFT_HIGH
            #pragma multi_compile_fragment _ _SCREEN_SPACE_OCCLUSION
            #pragma multi_compile_fragment _ _LIGHT_COOKIES
            #pragma multi_compile _ _LIGHT_LAYERS

            #pragma multi_compile_instancing
            #pragma instancing_options renderinglayer

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Fog.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/RenderingLayers.hlsl"

            TEXTURE2D(_BaseMap);   SAMPLER(sampler_BaseMap);
            TEXTURE2D(_NormalMap); SAMPLER(sampler_NormalMap);
            TEXTURE2D(_ORMMap);    SAMPLER(sampler_ORMMap);

            CBUFFER_START(UnityPerMaterial)
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
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                float4 tangentOS  : TANGENT;
                float2 uv         : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float2 uv          : TEXCOORD0;

                float3 positionWS  : TEXCOORD1;
                float3 normalWS    : TEXCOORD2;
                float4 tangentWS   : TEXCOORD3; // xyz tangent, w sign
                float3 viewDirWS   : TEXCOORD4;
                half3  vertexLight : TEXCOORD5;

                float4 shadowCoord : TEXCOORD6;

                float4 positionCS  : SV_POSITION;

                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            float2 TransformUV(float2 uv)
            {
                uv -= 0.5;
                float2x2 m = float2x2(_UVMatrix.x, _UVMatrix.y,
                                      _UVMatrix.z, _UVMatrix.w);
                uv = mul(m, uv);
                uv += 0.5;
                return uv * _UVScale.xy + _UVOffset.xy;
            }

            Varyings vert (Attributes v)
            {
                Varyings o = (Varyings)0;

                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                VertexPositionInputs posInputs = GetVertexPositionInputs(v.positionOS.xyz);
                VertexNormalInputs   norInputs = GetVertexNormalInputs(v.normalOS, v.tangentOS);

                o.positionCS = posInputs.positionCS;
                o.positionWS = posInputs.positionWS;

                o.normalWS = norInputs.normalWS;

                // Tangent with correct handedness/sign (URP style)
                real sign = v.tangentOS.w * GetOddNegativeScale();
                o.tangentWS = float4(norInputs.tangentWS.xyz, sign);

                o.viewDirWS = GetWorldSpaceViewDir(o.positionWS);

                // VertexLighting is used by URP’s InputData
                o.vertexLight = VertexLighting(o.positionWS, o.normalWS);

                // URP shadow coord path (works with screen-space shadows variants)
                o.shadowCoord = GetShadowCoord(posInputs);

                o.uv = TransformUV(v.uv);
                return o;
            }

            void InitializeInputDataLikeLit(Varyings i, half3 normalTS, out InputData inputData)
            {
                inputData = (InputData)0;

                inputData.positionWS = i.positionWS;

                half3 viewDirWS = SafeNormalize(i.viewDirWS);

                // Transform tangent-space normal -> world (URP lit style)
                float sgn = i.tangentWS.w;
                float3 bitangent = sgn * cross(i.normalWS, i.tangentWS.xyz);
                inputData.normalWS = TransformTangentToWorld(normalTS, half3x3(i.tangentWS.xyz, bitangent, i.normalWS));
                inputData.normalWS = NormalizeNormalPerPixel(inputData.normalWS);

                inputData.viewDirectionWS = viewDirWS;

                // Shadow coord & fog
                inputData.shadowCoord = i.shadowCoord;
                inputData.fogCoord = ComputeFogFactor(i.positionCS.z);

                // These 4 lines are the “missing contract” that commonly breaks lighting
                inputData.vertexLighting = i.vertexLight;
                inputData.bakedGI = SampleSH(inputData.normalWS);
                inputData.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(i.positionCS);
                inputData.shadowMask = half4(1,1,1,1); // if you don’t use lightmaps/shadowmask, keep it neutral
            }

            half4 frag (Varyings i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                // Tangent-space normal map input (as you noted)
                half3 normalTS = UnpackNormal(SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, i.uv));
                normalTS = lerp(normalTS, half3(0,0,1), 1.0 - _NormalStrength);

                // BaseColor grading (your Unreal logic)
                float3 baseColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv).rgb;
                baseColor = pow(max(baseColor * _ColorIntensity, 0.0001), _ColorPower);

                float luma = dot(baseColor, float3(0.2126, 0.7152, 0.0722));
                baseColor = lerp(baseColor, luma.xxx, _Desaturate);
                baseColor *= _Tint.rgb;

                // ORM unpack: R=AO, G=Roughness, B=Metallic
                float3 orm = SAMPLE_TEXTURE2D(_ORMMap, sampler_ORMMap, i.uv).rgb;

                float ao = orm.r;
                float roughness = pow(max(orm.g * _RoughnessScale, 0.0001), _RoughnessPower);
                float metallic  = pow(max(orm.b * _MetallicScale, 0.0001), _MetallicPower);
                float smoothness = saturate(1.0 - roughness);

                float3 emission = lerp(_EmissiveA.rgb, _EmissiveB.rgb, _EmissiveBlend);

                SurfaceData surface = (SurfaceData)0;
                surface.albedo     = baseColor;
                surface.metallic   = metallic;
                surface.smoothness = smoothness;
                surface.occlusion  = ao;
                surface.emission   = emission;
                surface.alpha      = 1.0;
                surface.normalTS   = normalTS; // IMPORTANT: feed TS normal so URP-style init can transform it

                InputData inputData;
                InitializeInputDataLikeLit(i, surface.normalTS, inputData);

                return UniversalFragmentPBR(inputData, surface);
            }

            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
