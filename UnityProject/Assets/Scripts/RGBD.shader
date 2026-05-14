Shader "Hidden/LinearDepth_HDRP"
{
    SubShader
    {
        Tags { "RenderPipeline" = "HDRenderPipeline" }
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            Name "LinearDepth"

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.high-definition/Runtime/ShaderLibrary/ShaderVariables.hlsl"

            // Full-screen triangle vertex input
            struct Attributes
            {
                uint vertexID : SV_VertexID;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 texcoord   : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                // HDRP provides a full-screen triangle helper
                output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
                output.texcoord   = GetFullScreenTriangleTexCoord(input.vertexID);

                return output;
            }

            float4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                float2 uv = input.texcoord;

                // HDRP renders with Y flipped relative to screen UV convention.
                // Flipping here corrects the upside-down result.
                uv.y = 1.0 - uv.y;

                // Sample raw (device/NDC) depth from the HDRP camera depth buffer.
                float rawDepth = SampleCameraDepth(uv);

                // Convert raw device depth to linear eye depth.
                // LinearEyeDepth(_ZBufferParams) already accounts for UNITY_REVERSED_Z
                // internally — do NOT manually flip rawDepth before passing it in.
                float linearDepth = LinearEyeDepth(rawDepth, _ZBufferParams);

                return float4(linearDepth, 0.0, 0.0, 1.0);
            }
            ENDHLSL
        }
    }
}
