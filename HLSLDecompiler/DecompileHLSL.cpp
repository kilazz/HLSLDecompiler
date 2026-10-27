// DecompileHLSL.cpp : Translates DirectX 11 DXBC assembly into valid, re-compilable HLSL.
// Target supports Vertex (VS), Pixel (PS), Geometry (GS), Hull (HS), Domain (DS), and Compute (CS) stages.
//
// The goal with this file is too keep it as much C++ as possible.
// So we avoid adding Windows specific stuff if at all possible.
//
// To that end, we are also changing some fundamental use types from int/long to size_t
// where it makes sense as an actual size measurement. We are doing this because the basic
// functions like strlen and sizeof return size_t, which varies in size from x32 to x64.
// Otherwise comparisons to strlen or assignments can report warnings of truncation.
// This also fixes all signed/unsigned mismatch warnings.
//
// Also using 'size_t' for some for loops here, because of the inherent problems of 'int'
// being signed. I tried using 'auto', but it made poor choices that still had warnings.
// I changed all the variants that are simple and clear, and left the unusual, less clear variants as is.
// Changing to size_t fixes a large number of signed/unsigned mismatch warnings.
//
// We are also using the CRT_SECURE flags to automatically fix sprintf, sscanf into _s safe versions
// where it can. _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1 _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_COUNT = 1
//
// All sscanf that did not use %s or %c were switched to sscanf_s to avoid warnings.

#include <map>
#include <string>
#include <cstdio>
#include <vector>
#include <set>
#include <algorithm>

#include "DecompileHLSL.h"
#include "BinaryDecompiler\include\pstdint.h"
#include "BinaryDecompiler\internal_includes\structs.h"
#include "BinaryDecompiler\internal_includes\decode.h"
#include <excpt.h>
#include "assert.h"
#include "log.h"
#include "version.h"

using namespace std;

// Supported internal data types mapped from constant buffer and signature declarations
enum DataType {
	DT_bool,
	DT_bool4,
	DT_float,
	DT_float2,
	DT_float3,
	DT_float4,
	DT_float4x2,
	DT_float3x3,
	DT_float4x3,
	DT_float2x4,
	DT_float3x4,
	DT_float4x4,
	DT_uint4,
	DT_int4,
	DT_uint3,
	DT_int3,
	DT_uint2,
	DT_int2,
	DT_uint,
	DT_int,
	DT_Unknown
};

// Represents an entry in a constant buffer
struct BufferEntry {
	DataType bt;
	int matrixRow;
	bool isRowMajor;
	string Name;
};

struct PatchConstantAssignment {
	string targetField;
	int slot;
	string mask;
};

struct CPOutputAssignment {
	string targetField;
	int slot;
	string mask;
};

// Map key is (register << 16) + byte_offset
typedef map<int, BufferEntry> CBufferData;
typedef map<string, string> StringStringMap;

// Convenience routine to calculate just the number of swizzle components.
// Used for ibfe. Inputs like 'o1.xy', return 2.
static string swizCount(const char *operand) {
	if (!operand)
		return "";
	const char *dot = strrchr(operand, '.');
	if (!dot)
		return "";
	size_t count = strlen(operand) - (dot - operand) - 1;
	return (count == 1) ? "" : to_string(count);
}

class Decompiler {
  public:
	// Constant buffer descriptor storage
	CBufferData mCBufferData;
	map<string, int> mCBufferNames;

	// Resource descriptor storage
	map<int, string> mSamplerNames;
	map<int, int> mSamplerNamesArraySize;
	map<int, string> mSamplerComparisonNames;
	map<int, int> mSamplerComparisonNamesArraySize;
	map<int, string> mTextureNames;
	map<int, int> mTextureNamesArraySize;
	map<int, string> mTextureType;

	// Register tracking and remapping
	map<string, string> mOutputRegisterValues;
	map<string, DataType> mOutputRegisterType;
	string mShaderType;
	string mSV_Position;
	bool mUsesProjection;
	Instruction *mLastStatement;
	string mMulOperand, mMulOperand2, mMulTarget;
	StringStringMap mCorrectedIndexRegisters;
	StringStringMap mRemappedOutputRegisters;
	vector<pair<string, string>> mRemappedInputRegisters;
	set<string> mBooleanRegisters;

	int StereoParamsReg;
	int IniParamsReg;

	// Main output buffer
	vector<char> mOutput;

	// Tessellation & Compute Metadata
	string mDomainType = "tri";
	string mPartitioning = "fractional_odd";
	string mOutputTopology = "triangle_cw";
	int mInputControlPoints = 3;
	int mOutputControlPoints = 3;
	float mMaxTessFactor = 64.0f;
	int mThreadGroup[3] = {1, 1, 1};

	// Hull Shader Phase Buffers
	int mHSPhase = 0; // 0 = Main Output, 1 = Control Point, 2 = Fork/Join
	vector<char> mHSControlPointPhase;
	vector<char> mHSForkPhase;
	set<int> mDeclaredTempsHS_Main;
	set<int> mDeclaredTempsHS_Fork;

	// Structs definitions generated from signatures
	string mStructInput;
	string mStructOutput;
	string mStructPatchConstants;
	string mDSUnpackCode;
	string mHSUnpackCode;
	map<int, string> mPatchConstantMapping;
	map<int, string> mCPOutputMapping;
	vector<PatchConstantAssignment> mHSPatchConstantAssignments;
	vector<CPOutputAssignment> mHSCPOutputAssignments;

	size_t mCodeStartPos;
	bool mErrorOccurred;
	bool mFixSvPosition;
	bool mRecompileVs;
	bool mPatched;
	string ZRepair_DepthTexture1, ZRepair_DepthTexture2;
	string BackProject_Vector1, BackProject_Vector2;
	char ZRepair_DepthTextureReg1, ZRepair_DepthTextureReg2;
	vector<string> ZRepair_Dependencies1, ZRepair_Dependencies2;
	bool mZRepair_DepthBuffer;
	vector<string> InvTransforms;
	string ZRepair_ZPosCalc1, ZRepair_ZPosCalc2;
	string ZRepair_PositionTexture;
	string ZRepair_WorldPosCalc;
	string ObjectPos_ID1, ObjectPos_ID2, ObjectPos_MUL1, ObjectPos_MUL2;
	string MatrixPos_ID1, MatrixPos_MUL1;
	int uuidVar;

	// Auto-indent of generated code
	const char *indent = "  ";
	int nestCount;

	// Statement operand buffers and pointer aliases
	char statement[128];
	char op1[opcodeSize], op2[opcodeSize], op3[opcodeSize], op4[opcodeSize], op5[opcodeSize], op6[opcodeSize], op7[opcodeSize],
		op8[opcodeSize], op9[opcodeSize], op10[opcodeSize], op11[opcodeSize], op12[opcodeSize], op13[opcodeSize], op14[opcodeSize],
		op15[opcodeSize];
	char *ops[15];

	// Suppress all these warnings, as they are an _int64 mismatch for the aging sscanf, which should
	// accept that in x64, but doesn't. Requiring int instead. Not worth altering for a benign warning.
#pragma warning(push)
#pragma warning(disable : 6328)

	Decompiler() : mLastStatement(0), uuidVar(0), nestCount(0) {
		ops[0] = op1;
		ops[1] = op2;
		ops[2] = op3;
		ops[3] = op4;
		ops[4] = op5;
		ops[5] = op6;
		ops[6] = op7;
		ops[7] = op8;
		ops[8] = op9;
		ops[9] = op10;
		ops[10] = op11;
		ops[11] = op12;
		ops[12] = op13;
		ops[13] = op14;
		ops[14] = op15;
	}

	void logDecompileError(const string &err) {
		// mErrorOccurred = true; // Kept disabled to allow fallback decompilation on unknown statements
		printf("    [DEBUG] Parser Warning: %s\n", err.c_str());
		LogInfo("    warning parsing shader> %s\n", err.c_str());
	}

	DataType TranslateType(const char *name) {
		if (!strcmp(name, "float4x4"))
			return DT_float4x4;
		if (!strcmp(name, "float4x3"))
			return DT_float4x3;
		if (!strcmp(name, "float4x2"))
			return DT_float4x2;
		if (!strcmp(name, "float2x4"))
			return DT_float2x4;
		if (!strcmp(name, "float3x4"))
			return DT_float3x4;
		if (!strcmp(name, "float4"))
			return DT_float4;
		if (!strcmp(name, "float3x3"))
			return DT_float3x3;
		if (!strcmp(name, "float3"))
			return DT_float3;
		if (!strcmp(name, "float2"))
			return DT_float2;
		if (!strcmp(name, "float1"))
			return DT_float;
		if (!strcmp(name, "float"))
			return DT_float;
		if (!strcmp(name, "bool"))
			return DT_bool;
		if (!strcmp(name, "bool4"))
			return DT_bool4;
		if (!strcmp(name, "uint4"))
			return DT_uint4;
		if (!strcmp(name, "uint3"))
			return DT_uint3;
		if (!strcmp(name, "uint2"))
			return DT_uint2;
		if (!strcmp(name, "uint"))
			return DT_uint;
		if (!strcmp(name, "int4"))
			return DT_int4;
		if (!strcmp(name, "int3"))
			return DT_int3;
		if (!strcmp(name, "int2"))
			return DT_int2;
		if (!strcmp(name, "int"))
			return DT_int;
		logDecompileError("Unknown data type: " + string(name));
		return DT_Unknown;
	}

	// Make this bump to new line slightly more clear by making it a convenience routine.
	void NextLine(const char *c, size_t &pos, size_t max) {
		while (c[pos] != 0x0a && pos < max)
			pos++;
		pos++;
	}

	// Just take the current input line, and copy it to the output.
	// This is used when we aren't sure what to do with something, and gives us the lines
	// in the output as ASM for reference. Specifically modifying the input pos.
	void ASMLineOut(const char *c, size_t &pos, size_t max) {
		char buffer[256];
		const char *startPos = c + pos;
		const char *eolPos = strchr(startPos, '\n');
		std::string line(startPos, eolPos);
		sprintf_s(buffer, 256, "%s\n", line.c_str());
		appendOutput(buffer);
	}

	// This is to fix a specific problem seen with Batman, where the fxc compiler
	// is aggressive and packs the input down to a single float4, where it should
	// be two float2's. This is going to fix only this specific case, because there
	// are other times when the packing is right.
	// We'll do this by reading ahead a full line, and seeing if the upcoming
	// sequence is of the form:
	// Name                 Index   Mask Register SysValue  Format   Used
	// -------------------- ----- ------ -------- -------- ------- ------
	// TEXCOORD                 0   xy          0     NONE   float   xy
	// TEXCOORD                 1   xy          1     NONE   float   xy
	//
	// The goal is to switch from outputing:
	//		float2 v0 : TEXCOORD0,
	//		float2 v1 : TEXCOORD1,
	// to:
	//		float4 v0 : TEXCOORD0,	// change packing to float4 if different texcoord
	//		float2 v1 : TEXCOORD1,
	//
	// This is only going to be done for TEXCOORD. I've looked at examples of
	// outputs like SV_TARGET, where fxc could have packed them, but it did not.
	// Adding the extra parts there creates spurious warnings, so for now,
	// we'll only do TEXCOORD as a known problem.
	// This does also unfortunately create warnings for the TEXCOORD outputs, but
	// I was unable to find any other way to avoid the fxc packing optimization.
	bool SkipPacking(const char *c, const map<string, DataType> &inUse) {
		char name[256], mask[16], sysvalue[16], format[16];
		int index, reg1, reg2;
		format[0] = 0;
		mask[0] = 0;
		size_t pos = 0;

		int numRead = sscanf_s(c + pos, "// %s %d %s %d %s %s", name, (int)sizeof(name), &index, mask, (int)sizeof(mask), &reg1, sysvalue,
							   (int)sizeof(sysvalue), format, (int)sizeof(format));
		if (numRead != 6)
			return false;
		name[sizeof(name) - 1] = '\0'; // Appease the static analysis gods
		if (strcmp(name, "TEXCOORD") != 0)
			return false;

		// Any v* register already active needs to be skipped, to allow the
		// normal packing to succeed. input:v1.xy + w1.zw. or output:o1.xy + p1.zw
		if (inUse.find(string("v" + to_string(reg1))) != inUse.end())
			return false;
		if (inUse.find(string("o" + to_string(reg1))) != inUse.end())
			return false;

		// skip pointer in order to parse next line
		while (c[pos] != 0x0a && pos < 200)
			pos++;
		pos++;

		numRead = sscanf_s(c + pos, "// %s %d %s %d %s %s", name, (int)sizeof(name), &index, mask, (int)sizeof(mask), &reg2, sysvalue,
						   (int)sizeof(sysvalue), format, (int)sizeof(format));
		if (numRead != 6)
			return false;
		name[sizeof(name) - 1] = '\0'; // Appease the static analysis gods
		if (strcmp(name, "TEXCOORD") != 0)
			return false;

		// The key aspect is whether we are supposed to use a different Register.
		if (reg1 == reg2)
			return false;

		return true;
	}

	// Looking up the interpolation values for a given input, like 'centroid'
	// or 'linear centroid'. These are not found in the text declaration for
	// Input Signature, so we are fetching them from the already parsed James-
	// Jones input. Otherwise we'd need to parse the dcl_input_ps phrases.
	string GetInterpolation(Shader *shader, const string &vRegister) {
		string interpolation = "";
		for (const auto &declaration : shader->psDecl) {
			if (declaration.eOpcode == OPCODE_DCL_INPUT_PS) {
				int regOut = stoi(vRegister.substr(1));
				uint32_t binReg = declaration.asOperands[0].ui32RegisterNumber;
				if (regOut == (int)binReg) {
					switch (declaration.value.eInterpolation) {
					case INTERPOLATION_CONSTANT:
						// The constant interpolation modifier in assembly means do not
						// interpolate, which in HLSL uses the nointerpolation modifier.
						// Seen in Arkham Knight
						interpolation = "nointerpolation ";
						break;
						// Let's skip adding this, as it's the default, and adds noise.
						// case INTERPOLATION_LINEAR:
						//	interpolation = "linear ";
						//	break;
					case INTERPOLATION_LINEAR_CENTROID: interpolation = "linear centroid "; break;
					case INTERPOLATION_LINEAR_NOPERSPECTIVE: interpolation = "linear noperspective "; break;
					case INTERPOLATION_LINEAR_NOPERSPECTIVE_CENTROID: interpolation = "linear noperspective centroid "; break;
					case INTERPOLATION_LINEAR_SAMPLE: interpolation = "linear sample "; break;
					case INTERPOLATION_LINEAR_NOPERSPECTIVE_SAMPLE: interpolation = "linear noperspective sample "; break;
					default: break;
					}
				}
			}
		}
		return interpolation;
	}

	// =========================================================================
	// Input Signature Parser
	//
	// Name                 Index   Mask Register SysValue  Format   Used
	// -------------------- ----- ------ -------- -------- ------- ------
	// TEXCOORD                 0   xy          0     NONE   float   xy
	// TEXCOORD                 1   xy          1     NONE   float   xy
	// COLOR                    3   xyz         2     NONE   float   xyz
	// =========================================================================
	void ParseInputSignature(Shader *shader, const char *c, size_t size) {
		// DataType is not used here, just a convenience for calling SkipPacking.
		map<string, DataType> usedInputRegisters;
		map<int, int> inputSlotCount; // FIX: Track uses of hardware register slot to handle >= 3 semantics
		mRemappedInputRegisters.clear();

		bool isHS = (mShaderType.substr(0, 3) == "hs_");
		bool isDS = (mShaderType.substr(0, 3) == "ds_");

		const char *headerid = "// Input signature:";
		size_t pos = 0;
		while (pos < size - strlen(headerid)) {
			if (!strncmp(c + pos, headerid, strlen(headerid)))
				break;
			else
				NextLine(c, pos, size);
		}

		if (pos < size - strlen(headerid)) {
			// Skip header
			for (int i = 0; i < 4; ++i)
				NextLine(c, pos, size);

			if (isHS || isDS)
				mStructInput = "struct " + (isHS ? string("HS_INPUT_CP") : string("DS_INPUT_CP")) + " {\n";
			else
				mOutput.insert(mOutput.end(), "\nvoid main(\n", "\nvoid main(\n" + 12);

			if (isDS) {
				mDSUnpackCode = "  float4 vicp[" + to_string(mInputControlPoints) +
								"][32];\n"
								"  [unroll] for (int _cp = 0; _cp < " +
								to_string(mInputControlPoints) +
								"; ++_cp) {\n"
								"    [unroll] for (int _j = 0; _j < 32; ++_j) vicp[_cp][_j] = 0;\n";
			} else if (isHS) {
				mHSUnpackCode = "  float4 v[" + to_string(mInputControlPoints) +
								"][32];\n"
								"  float4 vicp[" +
								to_string(mInputControlPoints) +
								"][32];\n"
								"  [unroll] for (int _cp = 0; _cp < " +
								to_string(mInputControlPoints) +
								"; ++_cp) {\n"
								"    [unroll] for (int _j = 0; _j < 32; ++_j) { v[_cp][_j] = 0; vicp[_cp][_j] = 0; }\n";
			}

			// Read list
			while (pos < size) {
				char name[256], mask[16], format[16], format2[16];
				int index, slot;
				format[0] = 0;
				mask[0] = 0;
				if (!strncmp(c + pos, "// no Input", strlen("// no Input")))
					break;

				int numRead = sscanf_s(c + pos, "// %s %d %s %d %s %s", name, (int)sizeof(name), &index, mask, (int)sizeof(mask), &slot,
									   format2, (int)sizeof(format2), format, (int)sizeof(format));
				if (numRead != 6)
					break;

				// Finish type
				if (SkipPacking(c + pos, usedInputRegisters))
					sprintf_s(format2, "%s%d", format, 4); // Force to float4
				else if (strlen(mask) > 1)
					sprintf_s(format2, "%s%d", format, (int)strlen(mask)); // e.g. float2
				else
					strcpy_s(format2, format);

				// Already used?
				int count = inputSlotCount[slot]++;
				char registerName[32];
				string regNameStr;

				if (count == 0) {
					sprintf_s(registerName, sizeof(registerName), "v%d", slot);
					regNameStr = registerName;
					usedInputRegisters[regNameStr] = TranslateType(format2);
				} else {
					char prefix[32];
					if (count == 1)
						sprintf_s(prefix, sizeof(prefix), "w%d", slot);
					else
						sprintf_s(prefix, sizeof(prefix), "w%d_%d", slot, count);

					string regNameWithDot = string(prefix) + ".";
					const char *INDEX_MASK = "xyzw";
					string newName = regNameWithDot;
					for (size_t j = 0; j < strlen(mask); ++j)
						newName.push_back(INDEX_MASK[j]);
					string origReg = string("v") + to_string(slot);
					mRemappedInputRegisters.push_back(pair<string, string>(origReg + "." + string(mask), newName));

					if (strlen(mask) > 1) {
						for (size_t j = 0; j < strlen(mask); ++j) {
							newName = regNameWithDot;
							newName.push_back(INDEX_MASK[j]);
							string oldName = origReg + ".";
							oldName.push_back(mask[j]);
							mRemappedInputRegisters.push_back(pair<string, string>(oldName, newName));
						}
					}
					regNameStr = prefix;
				}

				// Fetch interpolation modifiers like 'centroid'
				string modifier = GetInterpolation(shader, regNameStr);
				char buffer[256];

				if (isHS || isDS) {
					sprintf_s(buffer, "  %s%s %s : %s%d;\n", modifier.c_str(), format2, regNameStr.c_str(), name, index);
					mStructInput += buffer;

					char assignBuf[512];
					const char *msk = (mask[0] == '\0') ? "xyzw" : mask;
					if (isDS) {
						sprintf_s(assignBuf, "    vicp[_cp][%d].%s = vicp_patch[_cp].%s;\n", slot, msk, regNameStr.c_str());
						mDSUnpackCode += assignBuf;
					} else if (isHS) {
						sprintf_s(assignBuf, "    v[_cp][%d].%s = ip[_cp].%s;\n    vicp[_cp][%d].%s = ip[_cp].%s;\n", slot, msk,
								  regNameStr.c_str(), slot, msk, regNameStr.c_str());
						mHSUnpackCode += assignBuf;
					}
				} else {
					sprintf_s(buffer, "  %s%s %s : %s%d,\n", modifier.c_str(), format2, regNameStr.c_str(), name, index);
					mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
				}

				NextLine(c, pos, size);
				if (!strncmp(c + pos, "//\n", 3) || !strncmp(c + pos, "//\r", 3))
					break;
			}

			if (isHS || isDS)
				mStructInput += "};\n\n";

			if (isDS) {
				mDSUnpackCode += "  }\n";
			} else if (isHS) {
				mHSUnpackCode += "  }\n";
			}
		}
	}

	// =========================================================================
	// Patch Constant Signature Parser
	// =========================================================================
	void ParsePatchConstantSignature(const char *c, size_t size) {
		bool isHS = (mShaderType.substr(0, 3) == "hs_");
		bool isDS = (mShaderType.substr(0, 3) == "ds_");
		if (!isHS && !isDS)
			return;

		mHSPatchConstantAssignments.clear();

		const char *headerid = "// Patch Constant signature:";
		size_t pos = 0;
		while (pos < size - strlen(headerid)) {
			if (!strncmp(c + pos, headerid, strlen(headerid)))
				break;
			else
				NextLine(c, pos, size);
		}

		if (pos < size - strlen(headerid)) {
			for (int i = 0; i < 4; ++i)
				NextLine(c, pos, size);

			mStructPatchConstants = "struct " + (isHS ? string("HS_PATCH_CONSTANTS") : string("DS_PATCH_CONSTANTS")) + " {\n";

			if (mDomainType == "quad") {
				mStructPatchConstants += "  float edges[4] : SV_TessFactor;\n  float inside[2] : SV_InsideTessFactor;\n";
			} else if (mDomainType == "tri") {
				mStructPatchConstants += "  float edges[3] : SV_TessFactor;\n  float inside : SV_InsideTessFactor;\n";
			} else {
				mStructPatchConstants += "  float edges[2] : SV_TessFactor;\n";
			}

			string vpcUnpack = "  float4 vpc0=0, vpc1=0, vpc2=0, vpc3=0, vpc4=0, vpc5=0, vpc6=0, vpc7=0, vpc8=0, vpc9=0;\n";

			while (pos < size) {
				char name[256], mask[16], format[16], format2[16];
				int index, slot;
				format[0] = 0;
				mask[0] = 0;
				if (!strncmp(c + pos, "// no Patch Constant", strlen("// no Patch Constant")))
					break;

				int numRead = sscanf_s(c + pos, "// %s %d %s %d %s %s", name, (int)sizeof(name), &index, mask, (int)sizeof(mask), &slot,
									   format2, (int)sizeof(format2), format, (int)sizeof(format));
				if (numRead == 6) {
					if (strlen(mask) > 1)
						sprintf_s(format2, "%s%d", format, (int)strlen(mask));
					else
						strcpy_s(format2, format);

					char buffer[256];
					if (strcmp(name, "SV_TessFactor") != 0 && strcmp(name, "SV_InsideTessFactor") != 0) {
						sprintf_s(buffer, "  %s vpc%d : %s%d;\n", format2, slot, name, index);
						mStructPatchConstants += buffer;
						if (isHS) {
							mPatchConstantMapping[slot] = "vpc" + to_string(slot);
							PatchConstantAssignment pca;
							pca.targetField = "vpc" + to_string(slot);
							pca.slot = slot;
							pca.mask = (mask[0] == '\0') ? "xyzw" : mask;
							mHSPatchConstantAssignments.push_back(pca);
						} else if (isDS) {
							const char *msk = (mask[0] == '\0') ? "xyzw" : mask;
							char unpackBuf[256];
							sprintf_s(unpackBuf, "  vpc%d.%s = patchConsts.vpc%d;\n", slot, msk, slot);
							vpcUnpack += unpackBuf;
						}
					} else {
						if (isHS) {
							PatchConstantAssignment pca;
							if (!strcmp(name, "SV_TessFactor")) {
								mPatchConstantMapping[slot] = "edges[" + to_string(index) + "]";
								pca.targetField = "edges[" + to_string(index) + "]";
							} else if (mDomainType == "quad") {
								mPatchConstantMapping[slot] = "inside[" + to_string(index) + "]";
								pca.targetField = "inside[" + to_string(index) + "]";
							} else {
								mPatchConstantMapping[slot] = "inside";
								pca.targetField = "inside";
							}
							pca.slot = slot;
							pca.mask = (mask[0] == '\0') ? "x" : mask;
							mHSPatchConstantAssignments.push_back(pca);
						}
					}
				}
				NextLine(c, pos, size);
				if (!strncmp(c + pos, "//\n", 3) || !strncmp(c + pos, "//\r", 3))
					break;
			}
			mStructPatchConstants += "};\n\n";

			if (isDS)
				mDSUnpackCode += vpcUnpack + "\n";
		}
	}

	// =========================================================================
	// Output Signature Parser
	// =========================================================================
	void ParseOutputSignature(const char *c, size_t size) {
		mOutputRegisterType.clear();
		mRemappedOutputRegisters.clear();
		mSV_Position.clear();
		mHSCPOutputAssignments.clear();
		map<int, int> outputSlotCount; // FIX: Track uses of hardware register slot to handle >= 3 semantics

		bool isHS = (mShaderType.substr(0, 3) == "hs_");
		bool isDS = (mShaderType.substr(0, 3) == "ds_");

		const char *headerid = "// Output signature:";
		size_t pos = 0;
		while (pos < size - strlen(headerid)) {
			if (!strncmp(c + pos, headerid, strlen(headerid)))
				break;
			else
				NextLine(c, pos, size);
		}

		if (isDS) {
			mOutput.insert(mOutput.end(), mStructInput.begin(), mStructInput.end());
			mOutput.insert(mOutput.end(), mStructPatchConstants.begin(), mStructPatchConstants.end());

			string dsHeader = "[domain(\"" + mDomainType + "\")]\nvoid main(\n  DS_PATCH_CONSTANTS patchConsts,\n";
			if (mDomainType == "tri")
				dsHeader += "  float3 vDomain : SV_DomainLocation,\n";
			else
				dsHeader += "  float2 vDomain : SV_DomainLocation,\n";
			dsHeader += "  const OutputPatch<DS_INPUT_CP, " + to_string(mInputControlPoints) + "> vicp_patch,\n";
			mOutput.insert(mOutput.end(), dsHeader.c_str(), dsHeader.c_str() + dsHeader.length());
		}

		if (pos < size - strlen(headerid)) {
			for (int i = 0; i < 4; ++i)
				NextLine(c, pos, size);

			if (isHS)
				mStructOutput = "struct HS_OUTPUT_CP {\n";

			while (pos < size) {
				char name[256], mask[16], format[16], format2[16];
				int index, slot;
				format[0] = 0;
				mask[0] = 0;
				if (!strncmp(c + pos, "// no Output", strlen("// no Output")))
					break;

				int numRead = sscanf_s(c + pos, "// %s %d %s %d %s %s", name, (int)sizeof(name), &index, mask, (int)sizeof(mask), &slot,
									   format2, (int)sizeof(format2), format, (int)sizeof(format));
				if (numRead == 6) {
					if (SkipPacking(c + pos, mOutputRegisterType))
						sprintf_s(format2, "%s%d", format, 4); // force to float4
					else if (strlen(mask) > 1)
						sprintf_s(format2, "%s%d", format, (int)strlen(mask)); // e.g. float2
					else
						strcpy_s(format2, format);

					int count = outputSlotCount[slot]++;
					char registerName[32];
					string regNameStr;

					if (count == 0) {
						sprintf_s(registerName, sizeof(registerName), "o%d", slot);
						regNameStr = registerName;
					} else {
						char prefix[32];
						if (count == 1)
							sprintf_s(prefix, sizeof(prefix), "p%d", slot);
						else
							sprintf_s(prefix, sizeof(prefix), "p%d_%d", slot, count);

						string newName = string(prefix) + ".";
						const char *INDEX_MASK = "xyzw";
						for (size_t j = 0; j < strlen(mask); ++j)
							newName.push_back(INDEX_MASK[j]);
						string origReg = string("o") + to_string(slot);
						mRemappedOutputRegisters[origReg + "." + string(mask)] = newName;

						if (strlen(mask) > 1) {
							for (size_t j = 0; j < strlen(mask); ++j) {
								newName = string(prefix) + ".";
								newName.push_back(INDEX_MASK[j]);
								string oldName = origReg + ".";
								oldName.push_back(mask[j]);
								mRemappedOutputRegisters[oldName] = newName;
							}
						}
						regNameStr = prefix;
					}
					char buffer[256];
					if (isHS) {
						sprintf_s(buffer, "  %s %s : %s%d;\n", format2, regNameStr.c_str(), name, index);
						mStructOutput += buffer;
						mCPOutputMapping[slot] = regNameStr;

						CPOutputAssignment cpo;
						cpo.targetField = regNameStr;
						cpo.slot = slot;
						cpo.mask = (mask[0] == '\0') ? "xyzw" : mask;
						mHSCPOutputAssignments.push_back(cpo);
					} else {
						sprintf_s(buffer, "  out %s %s : %s%d,\n", format2, regNameStr.c_str(), name, index);
						mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
					}
					if (!strcmp(name, "SV_Position"))
						mSV_Position = regNameStr;
					mOutputRegisterType[regNameStr] = TranslateType(format2);
				} else if (numRead == 3) {
					char reg[64], sysValue[64], buffer[256];
					numRead = sscanf_s(c + pos, "// %s %d %s %s %s %s", name, (int)sizeof(name), &index, mask, (int)sizeof(mask), reg,
									   (int)sizeof(reg), sysValue, (int)sizeof(sysValue), format, (int)sizeof(format));

					if (isHS) {
						sprintf_s(buffer, "  %s %s : %s;\n", format, reg, name);
						mStructOutput += buffer;

						CPOutputAssignment cpo;
						cpo.targetField = reg;
						cpo.slot = parseResourceIndex(reg);
						cpo.mask = (mask[0] == '\0') ? "xyzw" : mask;
						mHSCPOutputAssignments.push_back(cpo);
					} else {
						sprintf_s(buffer, "  out %s %s : %s,\n", format, reg, name);
						mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
					}
				}
				if (numRead != 6 && numRead != 3)
					break;
				NextLine(c, pos, size);
				if (!strncmp(c + pos, "//\n", 3) || !strncmp(c + pos, "//\r", 3))
					break;
			}
		}

		if (isHS) {
			mStructOutput += "};\n\n";
		} else {
			mOutput.pop_back(); // Strip trailing '\n'
			mOutput.pop_back(); // Strip trailing ','
			const char *mainFooter = ")\n{\n  uint4 bitmask = 0, uiDest = 0;\n  float4 fDest = 0;\n";
			mOutput.insert(mOutput.end(), mainFooter, mainFooter + strlen(mainFooter));

			if (isDS) {
				mOutput.insert(mOutput.end(), mDSUnpackCode.c_str(), mDSUnpackCode.c_str() + mDSUnpackCode.length());
			}
		}
	}

	void WriteZeroOutputSignature(const char *c, size_t size) {
		const char *headerid = "// Output signature:";
		size_t pos = 0;
		while (pos < size - strlen(headerid)) {
			if (!strncmp(c + pos, headerid, strlen(headerid)))
				break;
			else
				NextLine(c, pos, size);
		}
		if (pos >= size - strlen(headerid))
			return;
		for (int i = 0; i < 4; ++i)
			NextLine(c, pos, size);

		map<int, int> zeroOutputSlotCount;
		while (pos < size) {
			char name[256], mask[16], format[16], format2[16];
			int index, slot;
			format[0] = 0;
			mask[0] = 0;
			if (!strncmp(c + pos, "// no Output", strlen("// no Output")))
				break;
			int numRead = sscanf_s(c + pos, "// %s %d %s %d %s %s", name, (int)sizeof(name), &index, mask, (int)sizeof(mask), &slot,
								   format2, (int)sizeof(format2), format, (int)sizeof(format));
			if (numRead == 6) {
				int count = zeroOutputSlotCount[slot]++;
				char registerName[32];
				if (count == 0)
					sprintf_s(registerName, sizeof(registerName), "o%d", slot);
				else if (count == 1)
					sprintf_s(registerName, sizeof(registerName), "p%d", slot);
				else
					sprintf_s(registerName, sizeof(registerName), "p%d_%d", slot, count);
				char buffer[256];
				sprintf_s(buffer, "  %s = 0;\n", registerName);
				mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
			} else if (numRead == 3) {
				char sysValue[64];
				sscanf_s(c + pos, "// %s %d %s %s %s %s", name, (int)sizeof(name), &index, mask, (int)sizeof(mask), sysValue,
						 (int)sizeof(sysValue), format2, (int)sizeof(format2), format, (int)sizeof(format));
				char buffer[256];
				sprintf_s(buffer, "  %s = 0;\n", sysValue);
				mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
			}
			NextLine(c, pos, size);
			if (!strncmp(c + pos, "//\n", 3) || !strncmp(c + pos, "//\r", 3))
				break;
		}
	}

	void ReadResourceBindings(const char *c, size_t size) {
		mCBufferNames.clear();
		mSamplerNames.clear();
		mSamplerNamesArraySize.clear();
		mSamplerComparisonNames.clear();
		mSamplerComparisonNamesArraySize.clear();
		mTextureNames.clear();
		mTextureNamesArraySize.clear();
		const char *headerid = "// Resource Bindings:";
		size_t pos = 0;
		while (pos < size - strlen(headerid)) {
			if (!strncmp(c + pos, headerid, strlen(headerid)))
				break;
			else
				NextLine(c, pos, size);
		}
		if (pos >= size - strlen(headerid))
			return;
		for (int i = 0; i < 4; ++i)
			NextLine(c, pos, size);

		while (pos < size) {
			char name[256], type[16], format[16], dim[16], bind[16];
			int arraySize;
			type[0] = 0;
			int numRead = sscanf_s(c + pos, "// %s %s %s %s %s %d", name, (int)sizeof(name), type, (int)sizeof(type), format,
								   (int)sizeof(format), dim, (int)sizeof(dim), bind, (int)sizeof(bind), &arraySize);

			if (numRead != 6) {
				logDecompileError("Error parsing resource declaration: " + string(c + pos, 80));
				NextLine(c, pos, size);
				continue;
			}

			int slot = 0;
			for (size_t i = 0; i < sizeof(bind) && bind[i]; ++i) {
				if (bind[i] >= '0' && bind[i] <= '9') {
					slot = atoi(&bind[i]);
					break;
				}
			}

			if (!strcmp(type, "sampler")) {
				char *escapePos = strchr(name, '[');
				if (escapePos)
					*escapePos = '_';
				escapePos = strchr(name, ']');
				if (escapePos)
					*escapePos = '_';
				string baseName = string(name) + "_s";
				mSamplerNames[slot] = baseName;
				mSamplerNamesArraySize[slot] = arraySize;
				if (arraySize > 1) {
					for (int i = 0; i < arraySize; ++i) {
						sprintf_s(name, "%s[%d]", baseName.c_str(), i);
						mSamplerNames[slot + i] = name;
					}
				}
			} else if (!strcmp(type, "sampler_c")) {
				char *escapePos = strchr(name, '[');
				if (escapePos)
					*escapePos = '_';
				escapePos = strchr(name, ']');
				if (escapePos)
					*escapePos = '_';
				string baseName = string(name) + "_s";
				mSamplerComparisonNames[slot] = baseName;
				mSamplerComparisonNamesArraySize[slot] = arraySize;
				if (arraySize > 1) {
					for (int i = 0; i < arraySize; ++i) {
						sprintf_s(name, "%s[%d]", baseName.c_str(), i);
						mSamplerComparisonNames[slot + i] = name;
					}
				}
			} else if (!strcmp(type, "texture")) {
				char *escapePos = strchr(name, '[');
				if (escapePos)
					*escapePos = '_';
				escapePos = strchr(name, ']');
				if (escapePos)
					*escapePos = '_';
				string baseName = string(name);
				mTextureNames[slot] = baseName;
				mTextureNamesArraySize[slot] = arraySize;
				if (arraySize > 1) {
					for (int i = 0; i < arraySize; ++i) {
						sprintf_s(name, "%s[%d]", baseName.c_str(), i);
						mTextureNames[slot + i] = name;
					}
				}
				if (!strcmp(dim, "1d"))
					mTextureType[slot] = "Texture1D<" + string(format) + ">";
				else if (!strcmp(dim, "2d"))
					mTextureType[slot] = "Texture2D<" + string(format) + ">";
				else if (!strcmp(dim, "2darray"))
					mTextureType[slot] = "Texture2DArray<" + string(format) + ">";
				else if (!strcmp(dim, "3d"))
					mTextureType[slot] = "Texture3D<" + string(format) + ">";
				else if (!strcmp(dim, "cube"))
					mTextureType[slot] = "TextureCube<" + string(format) + ">";
				else if (!strcmp(dim, "cubearray"))
					mTextureType[slot] = "TextureCubeArray<" + string(format) + ">";
				else if (!strncmp(dim, "2dMS", 4)) {
					// The documentation says it's not legal, but we see Texture 2DMS with no ending size in WatchDogs.
					// Instead of "2dMS4", it's just "2dMS". If we set that to Texture2DMS<float4>, with no size,
					// it generates the same code as the ASM.
					int msnumber, scanned;
					char buffer[256];
					scanned = sscanf_s(dim + 4, "%d", &msnumber);
					if (scanned == 1)
						sprintf_s(buffer, "Texture2DMS<%s,%d>", format, msnumber);
					else
						sprintf_s(buffer, "Texture2DMS<%s>", format);
					mTextureType[slot] = buffer;
				} else if (!strcmp(dim, "buf"))
					mTextureType[slot] = "Buffer<" + string(format) + ">";
				else if (!strcmp(dim, "r/o"))
					mTextureType[slot] = "StructuredBuffer<" + string(name) + ">";
				else
					logDecompileError("Unknown texture dimension: " + string(dim));
			} else if (!strcmp(type, "cbuffer")) {
				mCBufferNames[name] = slot;
			}
			NextLine(c, pos, size);
			if (!strncmp(c + pos, "//\n", 3) || !strncmp(c + pos, "//\r", 3))
				break;
		}
	}

	void WriteResourceDefinitions() {
		char buffer[256];
		_snprintf_s(buffer, 256, 256, "\n");
		mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));

		for (const auto &entry : mSamplerNames) {
			int slot = entry.first;
			int arrSize = mSamplerNamesArraySize[slot];
			if (arrSize == 1) {
				sprintf_s(buffer, "SamplerState %s : register(s%d);\n", entry.second.c_str(), slot);
				mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
			} else if (arrSize > 1) {
				string baseName = entry.second.substr(0, entry.second.find('['));
				sprintf_s(buffer, "SamplerState %s[%d] : register(s%d);\n", baseName.c_str(), arrSize, slot);
				mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
			}
		}
		for (const auto &entry : mSamplerComparisonNames) {
			int slot = entry.first;
			int arrSize = mSamplerComparisonNamesArraySize[slot];
			if (arrSize == 1) {
				sprintf_s(buffer, "SamplerComparisonState %s : register(s%d);\n", entry.second.c_str(), slot);
				mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
			} else if (arrSize > 1) {
				string baseName = entry.second.substr(0, entry.second.find('['));
				sprintf_s(buffer, "SamplerComparisonState %s[%d] : register(s%d);\n", baseName.c_str(), arrSize, slot);
				mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
			}
		}
		for (const auto &entry : mTextureNames) {
			int slot = entry.first;
			int arrSize = mTextureNamesArraySize[slot];
			if (arrSize == 1) {
				sprintf_s(buffer, "%s %s : register(t%d);\n", mTextureType[slot].c_str(), entry.second.c_str(), slot);
				mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
			} else if (arrSize > 1) {
				string baseName = entry.second.substr(0, entry.second.find('['));
				sprintf_s(buffer, "%s %s[%d] : register(t%d);\n", mTextureType[slot].c_str(), baseName.c_str(), arrSize, slot);
				mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
			}
		}
	}

	int getDataTypeSize(DataType d) {
		switch (d) {
		case DT_bool:
		case DT_uint:
		case DT_int:
		case DT_float: return 4;
		case DT_float2:
		case DT_uint2:
		case DT_int2: return 8;
		case DT_float3:
		case DT_uint3:
		case DT_int3: return 12;
		case DT_bool4:
		case DT_float4:
		case DT_uint4:
		case DT_int4: return 16;
		case DT_float2x4:
		case DT_float4x2: return 32;
		case DT_float3x3: return 36;
		case DT_float3x4: return 48;
		case DT_float4x3: return 48;
		case DT_float4x4: return 64;
		default: return 0;
		}
	}

	void ParseBufferDefinitions(Shader *shader, const char *c, size_t size) {
		mUsesProjection = false;
		mCBufferData.clear();
		BufferEntry immediateEntry;
		immediateEntry.Name = "icb[0]";
		immediateEntry.matrixRow = 0;
		immediateEntry.isRowMajor = false;
		immediateEntry.bt = DT_float4;
		mCBufferData[-1 << 16] = immediateEntry;
		vector<int> pendingStructAttributes[8];
		int structLevel = -1;
		const char *headerid = "// cbuffer ";
		size_t pos = 0;
		while (pos < size - strlen(headerid)) {
			while (pos < size - strlen(headerid)) {
				if (!strncmp(c + pos, headerid, strlen(headerid)))
					break;
				else
					NextLine(c, pos, size);
			}
			if (pos >= size - strlen(headerid))
				return;
			char name[256];
			int numRead = sscanf_s(c + pos, "// cbuffer %s", name, (int)sizeof(name));
			if (numRead != 1)
				return;
			NextLine(c, pos, size);
			NextLine(c, pos, size);
			map<string, int>::iterator i = mCBufferNames.find(name);
			if (i == mCBufferNames.end())
				return;
			const int bufferRegister = i->second;
			char buffer[256];
			if (name[0] == '$')
				name[0] = '_';
			_snprintf_s(buffer, 256, 256, "\ncbuffer %s : register(b%d)\n{\n", name, bufferRegister);
			mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));

			do {
				const char *eolPos = strchr(c + pos, '\n');
				memcpy(buffer, c + pos, eolPos - c - pos + 1);
				buffer[eolPos - c - pos + 1] = 0;
				if (strstr(buffer, " {\n")) {
					NextLine(c, pos, size);
					continue;
				}
				if (buffer[0] == '/' && buffer[1] == '/') {
					int ePos = 2;
					while (buffer[ePos] != 0 &&
						   (buffer[ePos] == ' ' || buffer[ePos] == '\t' || buffer[ePos] == '\n' || buffer[ePos] == '\r'))
						++ePos;
					if (!buffer[ePos]) {
						NextLine(c, pos, size);
						continue;
					}
				}
				if (strstr(buffer, " struct\n") || strstr(buffer, " struct ")) {
					++structLevel;
					mOutput.insert(mOutput.end(), '\n');
					for (int i = -1; i < structLevel; ++i) {
						mOutput.insert(mOutput.end(), ' ');
						mOutput.insert(mOutput.end(), ' ');
					}
					const char *structHeader = "struct\n";
					mOutput.insert(mOutput.end(), structHeader, structHeader + strlen(structHeader));
					for (int i = -1; i < structLevel; ++i) {
						mOutput.push_back(' ');
						mOutput.push_back(' ');
					}
					const char *structHeader2 = "{\n";
					mOutput.insert(mOutput.end(), structHeader2, structHeader2 + strlen(structHeader2));
					NextLine(c, pos, size);
					continue;
				}
				if (strstr(buffer, " }")) {
					while (c[pos] != '}')
						++pos;
					int bpos = 0;
					while (c[pos] != ';')
						buffer[bpos++] = c[pos++];
					string structName = string(buffer + 2, buffer + bpos) + ".";
					while (c[pos] != '/' && pos < size)
						pos++;
					int offset = 0;
					numRead = sscanf_s(c + pos, "// Offset: %d", &offset);
					if (numRead != 1)
						return;
					if (!structLevel)
						sprintf_s(buffer + bpos, sizeof(buffer) - bpos, " : packoffset(c%d);\n\n", offset / 16);
					else
						sprintf_s(buffer + bpos, sizeof(buffer) - bpos, ";\n\n");
					for (int i = -1; i < structLevel; ++i) {
						mOutput.push_back(' ');
						mOutput.push_back(' ');
					}
					mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));

					size_t arrayPos = structName.find('[');
					if (arrayPos == string::npos) {
						for (vector<int>::iterator i = pendingStructAttributes[structLevel].begin();
							 i != pendingStructAttributes[structLevel].end(); ++i) {
							mCBufferData[*i].Name = structName + mCBufferData[*i].Name;
							if (structLevel)
								pendingStructAttributes[structLevel - 1].push_back(*i);
						}
					} else {
						int arraySize;
						if (sscanf_s(structName.c_str() + arrayPos + 1, "%d", &arraySize) != 1) {
							logDecompileError("Error parsing array size: " + structName);
							NextLine(c, pos, size);
							continue;
						}
						structName = structName.substr(0, arrayPos);
						int structSize = 0;
						for (vector<int>::iterator j = pendingStructAttributes[structLevel].begin();
							 j != pendingStructAttributes[structLevel].end(); ++j)
							structSize += getDataTypeSize(mCBufferData[*j].bt);
						for (int i = arraySize - 1; i >= 0; --i) {
							sprintf_s(buffer, sizeof(buffer), "%s[%d].", structName.c_str(), i);
							for (vector<int>::iterator j = pendingStructAttributes[structLevel].begin();
								 j != pendingStructAttributes[structLevel].end(); ++j) {
								mCBufferData[*j + i * structSize].Name = buffer + mCBufferData[*j].Name;
								if (structLevel)
									pendingStructAttributes[structLevel - 1].push_back(*j + i * structSize);
							}
						}
					}
					pendingStructAttributes[structLevel].clear();
					NextLine(c, pos, size);
					--structLevel;
					continue;
				}
				char type[16];
				type[0] = 0;
				// With very long names, the disassembled code can have no space at the end, like:
				//   float2 __0RealtimeReflMul__1EnvCubeReflMul__2__3;// Offset:   96 Size:     8
				numRead = sscanf_s(c + pos, "// %s %[^;]", type, (int)sizeof(type), name, (int)sizeof(name));
				if (numRead != 2) {
					NextLine(c, pos, size);
					continue;
				}

				string modifier;
				BufferEntry e;
				e.isRowMajor = false;
				if (!strcmp(type, "row_major") || !strcmp(type, "column_major")) {
					e.isRowMajor = !strcmp(type, "row_major");
					modifier = type;
					modifier.push_back(' ');
					numRead =
						sscanf_s(c + pos, "// %s %s %[^;]", buffer, (int)sizeof(buffer), type, (int)sizeof(type), name, (int)sizeof(name));
					if (numRead != 3) {
						NextLine(c, pos, size);
						continue;
					}
				}
				pos += 2;
				while (c[pos] != '/' && pos < size)
					pos++;
				int offset = 0;
				numRead = sscanf_s(c + pos, "// Offset: %d", &offset);
				if (numRead != 1) {
					NextLine(c, pos, size);
					continue;
				}
				e.Name = name;
				e.matrixRow = 0;
				e.bt = TranslateType(type);

				// Uses projection matrix?
				if (e.bt == DT_float4x4 && e.Name.find_first_of("Projection") >= 0) {
					eolPos = strchr(c + pos, '\n');
					if (*(eolPos - 1) != ']')
						mUsesProjection = true;
				}
				// SR3 variant of projection matrix.
				if (e.bt == DT_float4x4 && e.Name.find_first_of("projTM") >= 0) {
					eolPos = strchr(c + pos, '\n');
					if (*(eolPos - 1) != ']')
						mUsesProjection = true;
				}

				// Look for array based elements. We need an mCBufferData element for each possible array element,
				// to match possible uses of those offsets in the ASM code.
				size_t ep = e.Name.find('[');
				if (ep != string::npos &&
					(e.bt == DT_bool || e.bt == DT_bool4 || e.bt == DT_float || e.bt == DT_float2 || e.bt == DT_float3 ||
					 e.bt == DT_float4 || e.bt == DT_uint || e.bt == DT_uint2 || e.bt == DT_uint3 || e.bt == DT_uint4 || e.bt == DT_int ||
					 e.bt == DT_int2 || e.bt == DT_int3 || e.bt == DT_int4 || e.bt == DT_float4x4 || e.bt == DT_float3x4 ||
					 e.bt == DT_float4x3 || e.bt == DT_float3x3 || e.bt == DT_float4x2 || e.bt == DT_float2x4)) {
					int numElements = 0;
					sscanf_s(e.Name.substr(ep + 1).c_str(), "%d", &numElements);
					string baseName = e.Name.substr(0, ep);
					int counter = 0;
					if (e.bt == DT_float || e.bt == DT_bool || e.bt == DT_uint || e.bt == DT_int)
						counter = 4;
					else if (e.bt == DT_float2 || e.bt == DT_uint2 || e.bt == DT_int2 || e.bt == DT_float3 || e.bt == DT_uint3 ||
							 e.bt == DT_int3 || e.bt == DT_bool4 || e.bt == DT_float4 || e.bt == DT_uint4 || e.bt == DT_int4)
						counter = 16; // FIX: In HLSL constant buffers, ALL vector array elements align to 16 bytes!
					else if (e.bt == DT_float4x2 || e.bt == DT_float2x4)
						counter = 16 * 2;
					else if (e.bt == DT_float3x3 || e.bt == DT_float4x3)
						counter = 16 * 3;
					else if (e.bt == DT_float3x4 || e.bt == DT_float4x4)
						counter = 16 * 4;
					else
						counter = 16;

					// Correct possible invalid array size. (16 byte boundaries)
					int byteSize;
					sscanf_s(strstr(c + pos, "Size:") + 5, "%d", &byteSize);
					if ((counter == 4 && numElements * counter < byteSize) || (counter == 12 && numElements * counter < byteSize))
						counter = 16;

					for (int i = 0; i < numElements; ++i) {
						sprintf_s(buffer, sizeof(buffer), "%s[%d]", baseName.c_str(), i);
						e.Name = buffer;
						int offsetPos;
						switch (e.bt) {
						case DT_float4x4:
						case DT_float3x4:
						case DT_float2x4:
							e.matrixRow = 3;
							offsetPos = (bufferRegister << 16) + offset + i * counter + 3 * 16;
							mCBufferData[offsetPos] = e;
							if (structLevel >= 0)
								pendingStructAttributes[structLevel].push_back(offsetPos);
						case DT_float4x3:
						case DT_float3x3:
							e.matrixRow = 2;
							offsetPos = (bufferRegister << 16) + offset + i * counter + 2 * 16;
							mCBufferData[offsetPos] = e;
							if (structLevel >= 0)
								pendingStructAttributes[structLevel].push_back(offsetPos);
						case DT_float4x2:
							e.matrixRow = 1;
							offsetPos = (bufferRegister << 16) + offset + i * counter + 1 * 16;
							mCBufferData[offsetPos] = e;
							if (structLevel >= 0)
								pendingStructAttributes[structLevel].push_back(offsetPos);
							e.matrixRow = 0;
							offsetPos = (bufferRegister << 16) + offset + i * counter;
							mCBufferData[offsetPos] = e;
							if (structLevel >= 0)
								pendingStructAttributes[structLevel].push_back(offsetPos);
							break;
						default:
							offsetPos = (bufferRegister << 16) + offset + i * counter;
							mCBufferData[offsetPos] = e;
							if (structLevel >= 0)
								pendingStructAttributes[structLevel].push_back(offsetPos);
						}
					}
				}
				// Non-array versions of floatYxZ
				else if (e.bt == DT_float4x2 || e.bt == DT_float3x3 || e.bt == DT_float4x3 || e.bt == DT_float2x4 || e.bt == DT_float3x4 ||
						 e.bt == DT_float4x4) {
					e.matrixRow = 0;
					int offsetPos = (bufferRegister << 16) + offset;
					mCBufferData[offsetPos] = e;
					if (structLevel >= 0)
						pendingStructAttributes[structLevel].push_back(offsetPos);
					e.matrixRow = 1;
					offsetPos = (bufferRegister << 16) + offset + 1 * 16;
					mCBufferData[offsetPos] = e;
					if (structLevel >= 0)
						pendingStructAttributes[structLevel].push_back(offsetPos);
					if (e.bt != DT_float4x2) {
						e.matrixRow = 2;
						offsetPos = (bufferRegister << 16) + offset + 2 * 16;
						mCBufferData[offsetPos] = e;
						if (structLevel >= 0)
							pendingStructAttributes[structLevel].push_back(offsetPos);
						if (e.bt != DT_float4x3 && e.bt != DT_float3x3) {
							e.matrixRow = 3;
							offsetPos = (bufferRegister << 16) + offset + 3 * 16;
							mCBufferData[offsetPos] = e;
							if (structLevel >= 0)
								pendingStructAttributes[structLevel].push_back(offsetPos);
						}
					}
				}
				// Non-array versions of scalar entities like float4
				else {
					int offsetPos = (bufferRegister << 16) + offset;
					mCBufferData[offsetPos] = e;
					if (structLevel >= 0)
						pendingStructAttributes[structLevel].push_back(offsetPos);
				}

				// Default value?
				NextLine(c, pos, size);
				const char *defaultid = "//      = ";
				int packoffset = offset / 16;
				int suboffset = (offset % 16) / 4;
				const char INDEX_MASK[] = "xyzw";
				string structSpacing;
				for (int i = -1; i < structLevel; ++i)
					structSpacing += "  ";

				if (!strncmp(c + pos, defaultid, strlen(defaultid))) {
					if (e.bt == DT_bool4) {
						logDecompileError("*** assignment to bool4, unknown syntax\n");
					}
					// For bool values, the usual conversion by %e creates QNAN, so handle them specifically. (e.g. = 0xffffffff)
					else if (e.bt == DT_bool) {
						unsigned int bHex = 0;
						numRead = sscanf_s(c + pos, "// = 0x%lx", &bHex);
						NextLine(c, pos, size);
						string bString = (bHex == 0) ? "false" : "true";
						if (structLevel < 0) {
							if (suboffset == 0)
								sprintf_s(buffer, "  %s%s %s : packoffset(c%d) = %s;\n", modifier.c_str(), type, name, packoffset,
										  bString.c_str());
							else
								sprintf_s(buffer, "  %s%s %s : packoffset(c%d.%c) = %s;\n", modifier.c_str(), type, name, packoffset,
										  INDEX_MASK[suboffset], bString.c_str());
						} else
							sprintf_s(buffer, "  %s%s%s %s = %s;\n", structSpacing.c_str(), modifier.c_str(), type, name, bString.c_str());
						mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
					}
					// Special int case, to avoid converting to float badly, creating #QNAN instead.
					else if (e.bt == DT_int || e.bt == DT_int2 || e.bt == DT_int3 || e.bt == DT_int4) {
						int in[4] = {0, 0, 0, 0};
						numRead = sscanf_s(c + pos, "// = %i %i %i %i", in + 0, in + 1, in + 2, in + 3);
						NextLine(c, pos, size);
						if (structLevel < 0) {
							if (suboffset == 0)
								sprintf_s(buffer, "  %s%s %s : packoffset(c%d) = {", modifier.c_str(), type, name, packoffset);
							else
								sprintf_s(buffer, "  %s%s %s : packoffset(c%d.%c) = {", modifier.c_str(), type, name, packoffset,
										  INDEX_MASK[suboffset]);
						} else
							sprintf_s(buffer, "  %s%s%s %s = {", structSpacing.c_str(), modifier.c_str(), type, name);
						mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
						for (int i = 0; i < numRead - 1; ++i) {
							sprintf_s(buffer, "%i,", in[i]);
							mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
						}
						sprintf_s(buffer, "%i};\n", in[numRead - 1]);
						mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
					} else if (e.bt == DT_float || e.bt == DT_float2 || e.bt == DT_float3 || e.bt == DT_float4) {
						float v[4] = {0, 0, 0, 0};
						numRead = sscanf_s(c + pos, "// = 0x%lx 0x%lx 0x%lx 0x%lx", (unsigned long *)v + 0, (unsigned long *)v + 1,
										   (unsigned long *)v + 2, (unsigned long *)v + 3);
						NextLine(c, pos, size);
						if (structLevel < 0) {
							if (suboffset == 0)
								sprintf_s(buffer, "  %s%s %s : packoffset(c%d) = {", modifier.c_str(), type, name, packoffset);
							else
								sprintf_s(buffer, "  %s%s %s : packoffset(c%d.%c) = {", modifier.c_str(), type, name, packoffset,
										  INDEX_MASK[suboffset]);
						} else
							sprintf_s(buffer, "  %s%s%s %s = {", structSpacing.c_str(), modifier.c_str(), type, name);
						mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
						for (int i = 0; i < numRead - 1; ++i) {
							sprintf_s(buffer, "%.9g,", v[i]);
							mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
						}
						sprintf_s(buffer, "%.9g};\n", v[numRead - 1]);
						mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
					}
					// Only 4x4 for now, known needed case.
					else if (e.bt == DT_float4x4) {
						float v[16];
						for (size_t i = 0; i < 4; i++) {
							numRead =
								sscanf_s(c + pos, "//%*[ =]0x%lx 0x%lx 0x%lx 0x%lx", (unsigned long *)&v[i * 4 + 0],
										 (unsigned long *)&v[i * 4 + 1], (unsigned long *)&v[i * 4 + 2], (unsigned long *)&v[i * 4 + 3]);
							if (numRead != 4) {
								logDecompileError("Default values for float4x4 not read correctly, n:" + numRead);
								break;
							}
							NextLine(c, pos, size);
						}
						if (structLevel < 0) {
							if (suboffset == 0)
								sprintf_s(buffer, "  %s%s %s : packoffset(c%d) = {", modifier.c_str(), type, name, packoffset);
							else
								sprintf_s(buffer, "  %s%s %s : packoffset(c%d.%c) = {", modifier.c_str(), type, name, packoffset,
										  INDEX_MASK[suboffset]);
						} else
							sprintf_s(buffer, "  %s%s%s %s = {", structSpacing.c_str(), modifier.c_str(), type, name);
						mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
						for (int i = 0; i < 16 - 1; ++i) {
							sprintf_s(buffer, "%.9g,", v[i]);
							mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
						}
						sprintf_s(buffer, "%.9g};\n", v[15]);
						mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
					}
					// If we don't know what the initializer is, output ASM for manual fix
					else {
						sprintf_s(buffer, "  %s%s %s : packoffset(c%d) = {", modifier.c_str(), type, name, packoffset);
						appendOutput(buffer);
						sprintf_s(buffer, "Unknown bad code for initializer (needs manual fix):\n");
						appendOutput(buffer);
						ASMLineOut(c, pos, size);
						NextLine(c, pos, size);
						const char *con = "//        0x";
						while (!strncmp(c + pos, con, strlen(con))) {
							ASMLineOut(c, pos, size);
							NextLine(c, pos, size);
						}
					}
				} else {
					if (structLevel < 0) {
						if (suboffset == 0)
							sprintf_s(buffer, "  %s%s %s : packoffset(c%d);\n", modifier.c_str(), type, name, packoffset);
						else
							sprintf_s(buffer, "  %s%s %s : packoffset(c%d.%c);\n", modifier.c_str(), type, name, packoffset,
									  INDEX_MASK[suboffset]);
					} else
						sprintf_s(buffer, "  %s%s%s %s;\n", structSpacing.c_str(), modifier.c_str(), type, name);
					mOutput.insert(mOutput.end(), buffer, buffer + strlen(buffer));
				}
			} while (strncmp(c + pos, "// }", 4));

			const char *endBuffer = "}\n";
			mOutput.insert(mOutput.end(), endBuffer, endBuffer + strlen(endBuffer));
		}
	}

	// =========================================================================
	// Swizzle and Operand Translator
	// =========================================================================
	void applySwizzle(const char *left, char *right, bool useInt = false) {
		char right2[opcodeSize];
		if (strlen(right) == 0) {
			logDecompileError("applySwizzle called with no second parameter: " + string(left));
			return;
		}
		if (right[strlen(right) - 1] == ',')
			right[strlen(right) - 1] = 0;

		// Strip sign and absolute, so they can be re-added at the end.
		bool absolute = false, negative = false;
		if (right[0] == '-') {
			negative = true;
			strcpy_s(right2, sizeof(right2), &right[1]);
			strcpy_s(right, opcodeSize, right2);
		}
		if (right[0] == '|') {
			absolute = true;
			strncpy_s(right2, sizeof(right2), &right[1], strlen(right) - 2);
			strcpy_s(right, opcodeSize, right2);
		}

		// Fairly bold change here- this fetches the source swizzle from 'left', and it previously would
		// find the first dot in the string. That's not right for left side array indices, so I changed it
		// to look for the far right dot instead. Should be correct, but this is used everywhere.
		const char *strPos = strrchr(left, '.') + 1;
		char idx[4] = {-1, -1, -1, -1};
		char map[4] = {3, 0, 1, 2};
		size_t pos = 0; // Used as index into string buffer
		if (strPos == (const char *)1)
			strPos = "x";
		while (*strPos && pos < 4)
			idx[pos++] = map[*strPos++ - 'w'];

		// literal?
		if (right[0] == 'l') {
			strPos = strchr(right, ',');
			if (!strPos) {
				strcpy_s(right2, right + 2);
				right2[strlen(right2) - 1] = 0;
			} else {
				char *beginPos = right + 2;
				float args[4];
				for (int i = 0; i < 4; ++i) {
					char *endPos = strchr(beginPos, ',');
					if (endPos)
						*endPos = 0;
					sscanf_s(beginPos, "%f", args + i);
					beginPos = endPos + 1;
				}
				if (pos == 1) {
					sprintf_s(right2, "%.9g", args[idx[0]]);
				} else {
					// Only integer values?
					bool isInt = true;
					for (int i = 0; idx[i] >= 0 && i < 4; ++i)
						isInt = isInt && (floor(args[idx[i]]) == args[idx[i]]);
					if (isInt && useInt) {
						sprintf_s(right2, "int%Id(", pos);
						for (int i = 0; idx[i] >= 0 && i < 4; ++i)
							sprintf_s(right2 + strlen(right2), sizeof(right2) - strlen(right2), "%d,", int(args[idx[i]]));
						right2[strlen(right2) - 1] = 0;
						strcat_s(right2, ")");
					} else {
						sprintf_s(right2, "float%Id(", pos);
						for (int i = 0; idx[i] >= 0 && i < 4; ++i)
							sprintf_s(right2 + strlen(right2), sizeof(right2) - strlen(right2), "%.9g,", args[idx[i]]);
						right2[strlen(right2) - 1] = 0;
						strcat_s(right2, ")");
					}
				}
			}
		} else {
			strPos = strrchr(right, '.') + 1;
			if (strPos == (const char *)1) {
				logDecompileError("applySwizzle 2nd parameter missing '.': " + string(right));
				return;
			}
			strncpy_s(right2, right, strPos - right);
			right2[strPos - right] = 0;
			pos = strlen(right2);
			if (strlen(right) - strlen(right2) == 1)
				strcpy_s(right2, right);
			else {
				for (int i = 0; idx[i] >= 0 && i < 4; ++i)
					right2[pos++] = strPos[idx[i]];
				right2[pos] = 0;
			}

			// Buffer reference?
			// This section was changed to use sscanf_s as a more reliable way of parsing.
			strPos = strstr(right2, "cb");
			if (!strPos) {
				// SM5.1 / RenderDoc style disassembly uses upper-case CB# for constant buffers.
				// Normalise it back to the lower-case cb# via non-const pointer to prevent C3892.
				char *cbUpper = strstr(right2, "CB");
				if (cbUpper) {
					cbUpper[0] = 'c';
					cbUpper[1] = 'b';
					strPos = cbUpper;
				}
			}
			if (strPos) {
				int bufIndex = 0, bufOffset;
				char regAndSwiz[opcodeSize];

				// By scanning these in this order, we are sure to cover every variant, without mismatches.
				// Like: -cb2[r12.w+63].xyzx as : -cb(bufIndex)[(regAndSwiz)+(bufOffset)]
				if (sscanf_s(strPos, "cb%d[%[^+]+%d]", &bufIndex, regAndSwiz, (int)sizeof(regAndSwiz), &bufOffset) == 3) {
					CBufferData::iterator it = mCBufferData.find((bufIndex << 16) + bufOffset * 16);
					if (it == mCBufferData.end()) {
						logDecompileError("Missing constant buffer name: " + string(right2));
						return;
					}
					if (strncmp(strPos, it->second.Name.c_str(), 4) == 0) {
						string full = strPos;
						size_t dotspot = full.rfind('.');
						it->second.Name = full.substr(0, dotspot);
						regAndSwiz[0] = 0;
					}
				}
				// Like: cb1[29].xyzx Most common variant.
				else if (sscanf_s(strPos, "cb%d[%d]", &bufIndex, &bufOffset) == 2) {
					regAndSwiz[0] = 0;
				}
				// Like: cb0[r0.w].xy
				else if (sscanf_s(strPos, "cb%d[%s]", &bufIndex, regAndSwiz, (int)sizeof(regAndSwiz)) == 2) {
					bufOffset = 0;
				}
				// Like: icb[r0.w+0].xyzw
				else if (sscanf_s(strPos, "cb[%[^+]+%d]", regAndSwiz, (int)sizeof(regAndSwiz), &bufOffset) == 2) {
					bufIndex = -1;
				}
				// Like: icb[22].w
				else if (sscanf_s(strPos, "cb[%d]", &bufOffset) == 1) {
					bufIndex = -1;
					regAndSwiz[0] = 0;
				}
				// Like: icb[r1.z].xy
				else if (sscanf_s(strPos, "cb[%s]", regAndSwiz, (int)sizeof(regAndSwiz)) == 1) {
					bufIndex = -1;
					bufOffset = 0;
				} else {
					logDecompileError("Error parsing buffer register, unknown variant: " + string(right2));
					return;
				}

				// Missing mCBufferData name for this buffer entry. In the icb case, that's expected.
				if (bufIndex == -1) {
					BufferEntry immediateEntry;
					string full = right2;
					size_t dotspot = full.rfind('.');
					immediateEntry.Name = full.substr(0, dotspot);
					immediateEntry.matrixRow = 0;
					immediateEntry.isRowMajor = false;
					immediateEntry.bt = DT_float4;
					mCBufferData[(bufIndex << 16) + bufOffset * 16] = immediateEntry;
					regAndSwiz[0] = 0;
				}

				// Multi-component resolution for packed cbuffers across member boundaries.
				char *dotPtr = strrchr(right2, '.');
				const char *swizChars = dotPtr ? (dotPtr + 1) : "x";
				size_t numComps = strlen(swizChars);
				bool validSwizzle = (numComps >= 1 && numComps <= 4);
				if (validSwizzle) {
					for (size_t c = 0; c < numComps; ++c) {
						char ch = swizChars[c];
						if (ch != 'x' && ch != 'y' && ch != 'z' && ch != 'w') {
							validSwizzle = false;
							break;
						}
					}
				}

				auto isMatrixType = [](DataType bt) -> bool {
					return bt == DT_float4x4 || bt == DT_float4x3 || bt == DT_float4x2 || bt == DT_float2x4 || bt == DT_float3x4 ||
						   bt == DT_float3x3;
				};

				auto findVarForComp = [&](char compChar, string &outVarName, char &outCompChar, DataType &outType) -> bool {
					if (bufIndex < 0)
						return false;
					int subOff = (compChar == 'x') ? 0 : (compChar == 'y') ? 1 : (compChar == 'z') ? 2 : (compChar == 'w') ? 3 : -1;
					if (subOff < 0)
						return false;
					int targetByte = (bufIndex << 16) + bufOffset * 16 + subOff * 4;
					CBufferData::iterator it = mCBufferData.upper_bound(targetByte);
					if (it == mCBufferData.begin())
						return false;
					--it;
					if (it->first < 0 || (it->first >> 16) != bufIndex)
						return false;
					if (isMatrixType(it->second.bt))
						return false;
					int varStart = it->first;
					int diff = (targetByte - varStart) / 4;
					int varSize = getDataTypeSize(it->second.bt) / 4;
					if (diff < 0 || diff >= varSize || diff >= 4)
						return false;
					const char *MASK = "xyzw";
					outVarName = it->second.Name;
					outCompChar = MASK[diff];
					outType = it->second.bt;
					return true;
				};

				bool multiVariable = false;
				string compVarNames[4];
				char compSubSwiz[4];
				DataType compTypes[4];
				bool resolvedMulti = false;

				if (!regAndSwiz[0] && validSwizzle && bufIndex >= 0) {
					bool allFound = true;
					for (size_t c = 0; c < numComps; ++c) {
						if (!findVarForComp(swizChars[c], compVarNames[c], compSubSwiz[c], compTypes[c])) {
							allFound = false;
							break;
						}
						if (c > 0 && compVarNames[c] != compVarNames[0])
							multiVariable = true;
					}
					if (allFound) {
						char resolvedBuf[1024];
						if (!multiVariable) {
							DataType dt = compTypes[0];
							bool isScalar = (dt == DT_float || dt == DT_uint || dt == DT_int || dt == DT_bool);
							if (isScalar) {
								if (numComps == 1) {
									sprintf_s(resolvedBuf, "%s", compVarNames[0].c_str());
								} else {
									sprintf_s(resolvedBuf, "float%zu(", numComps);
									for (size_t c = 0; c < numComps; ++c)
										sprintf_s(resolvedBuf + strlen(resolvedBuf), sizeof(resolvedBuf) - strlen(resolvedBuf), "%s,",
												  compVarNames[0].c_str());
									resolvedBuf[strlen(resolvedBuf) - 1] = ')';
								}
							} else {
								string relSwiz = "";
								for (size_t c = 0; c < numComps; ++c)
									relSwiz += compSubSwiz[c];
								sprintf_s(resolvedBuf, "%s.%s", compVarNames[0].c_str(), relSwiz.c_str());
							}
						} else {
							sprintf_s(resolvedBuf, "float%zu(", numComps);
							for (size_t c = 0; c < numComps; ++c) {
								DataType dt = compTypes[c];
								bool isScalar = (dt == DT_float || dt == DT_uint || dt == DT_int || dt == DT_bool);
								if (isScalar)
									sprintf_s(resolvedBuf + strlen(resolvedBuf), sizeof(resolvedBuf) - strlen(resolvedBuf), "%s,",
											  compVarNames[c].c_str());
								else
									sprintf_s(resolvedBuf + strlen(resolvedBuf), sizeof(resolvedBuf) - strlen(resolvedBuf), "%s.%c,",
											  compVarNames[c].c_str(), compSubSwiz[c]);
							}
							resolvedBuf[strlen(resolvedBuf) - 1] = ')';
						}
						strcpy_s(right2, resolvedBuf);
						resolvedMulti = true;
					}
				}

				if (!resolvedMulti) {
					CBufferData::iterator i = mCBufferData.find((bufIndex << 16) + bufOffset * 16);
					if (i == mCBufferData.end() && strrchr(right2, '.')[1] == 'y')
						i = mCBufferData.find((bufIndex << 16) + bufOffset * 16 + 4);
					if (i == mCBufferData.end() && strrchr(right2, '.')[1] == 'z')
						i = mCBufferData.find((bufIndex << 16) + bufOffset * 16 + 8);
					if (i == mCBufferData.end() && strrchr(right2, '.')[1] == 'w')
						i = mCBufferData.find((bufIndex << 16) + bufOffset * 16 + 12);
					if (i == mCBufferData.end()) {
						logDecompileError("3 Error parsing buffer offset: " + string(right2));
						return;
					}

					char right3[opcodeSize];
					right3[0] = 0;
					strcat_s(right3, i->second.Name.c_str());
					strPos = strchr(strPos, ']');
					if (!strPos) {
						logDecompileError("4 Error parsing buffer offset: " + string(right2));
						return;
					}

					if (regAndSwiz[0]) {
						char *indexPos = strchr(right3, '[');
						if (indexPos)
							*indexPos = 0;
						string indexRegisterName(regAndSwiz, strchr(regAndSwiz, '.'));
						StringStringMap::iterator isCorrected = mCorrectedIndexRegisters.find(indexRegisterName);
						if (isCorrected != mCorrectedIndexRegisters.end()) {
							char newOperand[opcodeSize];
							strcpy_s(newOperand, isCorrected->second.c_str());
							applySwizzle(regAndSwiz, newOperand, true);
							sprintf_s(right3 + strlen(right3), sizeof(right3) - strlen(right3), "[%s]", newOperand);
						} else if (mLastStatement && mLastStatement->eOpcode == OPCODE_IMUL &&
								   (i->second.bt == DT_float4x4 || i->second.bt == DT_float4x3 || i->second.bt == DT_float4x2 ||
									i->second.bt == DT_float2x4 || i->second.bt == DT_float3x4 || i->second.bt == DT_float3x3)) {
							char newOperand[opcodeSize];
							strcpy_s(newOperand, mMulOperand.c_str());
							applySwizzle(regAndSwiz, newOperand, true);
							sprintf_s(right3 + strlen(right3), sizeof(right3) - strlen(right3), "[%s]", newOperand);
							mCorrectedIndexRegisters[indexRegisterName] = mMulOperand;
						} else if (i->second.bt == DT_float4x2 || i->second.bt == DT_float2x4)
							sprintf_s(right3 + strlen(right3), sizeof(right3) - strlen(right3), "[%s/2]", regAndSwiz);
						else if (i->second.bt == DT_float4x3 || i->second.bt == DT_float3x4)
							sprintf_s(right3 + strlen(right3), sizeof(right3) - strlen(right3), "[%s/3]", regAndSwiz);
						else if (i->second.bt == DT_float4x4)
							sprintf_s(right3 + strlen(right3), sizeof(right3) - strlen(right3), "[%s/4]", regAndSwiz);
						else {
							// Most common case, like: g_AmbientCube[r3.w] and Globals[r19.w+63]
							// Bug was to not handle the struct case here, and truncate string.
							// Like g_OmniLights[r5.w].m_PositionFar -> g_OmniLights[r5.w]
							// Start fresh with original string and just replace, not char* manipulate.
							// base e.g: g_OmniLights[0].m_PositionFar
							string base = i->second.Name;
							size_t left = base.find('[') + 1;
							size_t length = base.find(']') - left;

							// Another bug was to miss array indexing into a variable name. So if i->second.Name
							// is not a zero index element, we need to preserve the offset.
							// Like a JC3 variant of cb0[r19.w + 63].xyzw becoming Globals[r19.w+63]
							// Normally an offset turns into an actual named variable.
							int offset;
							sscanf_s(i->second.Name.c_str(), "%*[^[][%d]", &offset);
							if (offset == 0) {
								// e.g. cb0[r21.y + 55] -> PointLight[r21.y]
								base.replace(left, length, regAndSwiz);
							} else {
								// leave it named with register and bufOffset, e.g. cb0[r19.w + 63] -> Globals[r19.w+63]
								char regAndOffset[opcodeSize];
								sprintf_s(regAndOffset, "%s+%d", regAndSwiz, bufOffset);
								base.replace(left, length, regAndOffset);
							}
							strcpy_s(right3, base.c_str());
						}
					}
					if (i->second.bt != DT_float && i->second.bt != DT_bool && i->second.bt != DT_uint && i->second.bt != DT_int) {
						strcat_s(right3, ".");
						if (i->second.bt == DT_float4x4 || i->second.bt == DT_float4x3 || i->second.bt == DT_float4x2 ||
							i->second.bt == DT_float2x4 || i->second.bt == DT_float3x4 || i->second.bt == DT_float3x3) {
							strPos = strrchr(right2, '.');
							while (*++strPos) {
								switch (*strPos) {
								case 'x':
									sprintf_s(right3 + strlen(right3), sizeof(right3) - strlen(right3),
											  i->second.isRowMajor ? "_m%d0" : "_m0%d", i->second.matrixRow);
									break;
								case 'y':
									sprintf_s(right3 + strlen(right3), sizeof(right3) - strlen(right3),
											  i->second.isRowMajor ? "_m%d1" : "_m1%d", i->second.matrixRow);
									break;
								case 'z':
									sprintf_s(right3 + strlen(right3), sizeof(right3) - strlen(right3),
											  i->second.isRowMajor ? "_m%d2" : "_m2%d", i->second.matrixRow);
									break;
								case 'w':
									sprintf_s(right3 + strlen(right3), sizeof(right3) - strlen(right3),
											  i->second.isRowMajor ? "_m%d3" : "_m3%d", i->second.matrixRow);
									break;
								default: logDecompileError("Error parsing matrix index: " + string(right2));
								}
							}
						} else {
							strPos = strrchr(right2, '.');
							strcat_s(right3, strPos + 1);
						}
					}
					strcpy_s(right2, right3);
				}
			}
		}
		if (absolute && negative)
			sprintf_s(right, opcodeSize, "-abs(%s)", right2);
		else if (absolute)
			sprintf_s(right, opcodeSize, "abs(%s)", right2);
		else if (negative)
			sprintf_s(right, opcodeSize, "-%s", right2);
		else
			strcpy_s(right, opcodeSize, right2);
	}

	int ReadStatement(const char *pos) {
		// Kill newline.
		char lineBuffer[2048];
		strncpy_s(lineBuffer, pos, 2047);
		lineBuffer[2047] = 0;
		char *newlinePos = strchr(lineBuffer, '\n');
		if (newlinePos)
			*newlinePos = 0;

		statement[0] = 0;
		for (int i = 0; i < 15; ++i)
			ops[i][0] = 0;

		int numRead = sscanf_s(lineBuffer, "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s", statement, (int)sizeof(statement), op1,
							   opcodeSize, op2, opcodeSize, op3, opcodeSize, op4, opcodeSize, op5, opcodeSize, op6, opcodeSize, op7,
							   opcodeSize, op8, opcodeSize, op9, opcodeSize, op10, opcodeSize, op11, opcodeSize, op12, opcodeSize, op13,
							   opcodeSize, op14, opcodeSize, op15, opcodeSize);

		// Cull the [precise] from any instruction using it by moving down all opcodes.
		if (!strncmp(ops[0], "[precise", strlen("[precise"))) {
			for (int i = 0; i < 14; ++i)
				strcpy_s(ops[i], opcodeSize, ops[i + 1]);
			ops[14][0] = 0;
		}

		// Reassembles broken l(...) arguments that had spaces.
		for (int i = 0; i < 7; ++i) {
			size_t len = strlen(ops[i]);
			if (!strncmp(ops[i], "l(", 2) && (len < 2 || (ops[i][len - 1] != ')' && ops[i][len - 2] != ')'))) {
				strcat_s(ops[i], opcodeSize, " ");
				strcat_s(ops[i], opcodeSize, ops[i + 1]);
				strcat_s(ops[i], opcodeSize, " ");
				strcat_s(ops[i], opcodeSize, ops[i + 2]);
				strcat_s(ops[i], opcodeSize, " ");
				strcat_s(ops[i], opcodeSize, ops[i + 3]);
				for (int j = i + 1; j + 3 < 15; ++j)
					strcpy_s(ops[j], opcodeSize, ops[j + 3]);
				for (int j = 15 - 3; j < 15; ++j)
					ops[j][0] = 0;
			}
		}

		// Consolidate '+' tokens into operand additions
		for (int i = 0; i < 5; ++i) {
			while (!strcmp(ops[i + 1], "+")) {
				strcat_s(ops[i], opcodeSize, ops[i + 1]);
				strcat_s(ops[i], opcodeSize, ops[i + 2]);
				for (int j = i + 1; j + 2 < 15; ++j)
					strcpy_s(ops[j], opcodeSize, ops[j + 2]);
				ops[13][0] = 0;
				ops[14][0] = 0;
			}
		}

		return numRead;
	}

	string replaceInt(string input) {
		float number;
		if (sscanf_s(input.c_str(), "%f", &number) != 1)
			return input;
		if (floor(number) != number)
			return input;
		char buffer[64];
		sprintf_s(buffer, "%d", (int)number);
		return buffer;
	}

	string GetSuffix(char *s, int index) {
		if (s[strlen(s) - 1] == ',')
			s[strlen(s) - 1] = 0;
		char *dotPos = strrchr(s, '.');
		// Single byte?
		if (dotPos && dotPos[1] >= 'w' && dotPos[1] <= 'z') {
			string buffer = s;
			size_t targetPos = buffer.find_last_of('.');
			buffer[targetPos + 1] = dotPos[1 + index];
			buffer.resize(targetPos + 2);
			if (!buffer.substr(0, 4).compare("abs("))
				buffer.push_back(')');
			return buffer;
		}
		// Single literal?
		if (!strchr(s, ',')) {
			return s;
		}
		// Vector literal.
		if (index == 0) {
			string buffer = strchr(s, '(') + 1;
			buffer.resize(buffer.find(','));
			return replaceInt(buffer);
		}
		if (index == 1) {
			string buffer = strchr(s, ',') + 1;
			buffer.resize(buffer.find_first_of(",)"));
			return replaceInt(buffer);
		}
		if (index == 2) {
			dotPos = strchr(s, ',') + 1;
			char *endPos = strchr(dotPos, ',');
			if (!endPos)
				endPos = strchr(dotPos, ')');
			string buffer = endPos + 1;
			buffer.resize(buffer.find_first_of(",)"));
			return replaceInt(buffer);
		}
		string buffer = strrchr(s, ',') + 1;
		buffer.resize(buffer.rfind(')'));
		return replaceInt(buffer);
	}

	// Shader model 5.1 (and tools such as RenderDoc) emit resource operands using
	// upper-case logical names like T0, S0, U0, CB0, while SM4/SM5.0 fxc disassembly
	// uses lower-case t0, s0, u0, cb0. This helper pulls the numeric register/logical
	// index out of either form by skipping any leading resource-type letters, so the
	// rest of the decompiler can stay agnostic about the casing.
	static int parseResourceIndex(const char *op) {
		if (op == NULL)
			return 0;
		while (*op && (*op < '0' || *op > '9'))
			++op;
		return atoi(op);
	}

	// This was expected to take input of the form
	//	r1.xyxx
	// and decide how to truncate it based on the input type of Texture2D, 3D or Cube.
	// If it's Texture2D for example, that would trim to only the first two elements
	// as a 2D texture, so r1.xy
	//
	// This failed for constants of the form
	//	float4(0.5, 0.5, 0, 0)
	// which were found in AC3.
	// The fix is to look for that constant format too, and if we see those, truncate
	// the constant to match. So, for example, float4(0.5, 0.5, 0, 0) with Texture2D
	// becomes float2(0.5, 0.5) as the two elements.
	void truncateTexturePos(char *op, const char *textype) {
		char *cpos;
		if (!strncmp(op, "float", 5)) {
			if (!strncmp(textype, "Texture2D", 9)) {
				cpos = strchr(op, ',');
				cpos++;
				cpos = strchr(cpos, ',');
				cpos[0] = ')';
				cpos[1] = 0;
				op[5] = '2'; // now: float2(x,y)
			} else if (!strncmp(textype, "Texture3D", 9)) {
				cpos = strchr(op, ',');
				cpos++;
				cpos = strchr(cpos, ',');
				cpos++;
				cpos = strchr(cpos, ',');
				cpos[0] = ')';
				cpos[1] = 0;
				op[5] = '3'; // now: float3(x,y,z)
			}
		} else {
			// Normal variant like r1.xyxx
			int pos = 5;
			if (!strncmp(textype, "Texture1D<", strlen("Texture1D<")))
				pos = 2; // float .x
			else if (!strncmp(textype, "Texture2DMS<", strlen("Texture2DMS<")))
				pos = 3; // float2 .xy
			else if (!strncmp(textype, "Texture1DArray<", strlen("Texture1DArray<")))
				pos = 3;
			else if (!strncmp(textype, "Texture2D<", strlen("Texture2D<")))
				pos = 3;
			else if (!strncmp(textype, "Texture2DMSArray<", strlen("Texture2DMSArray<")))
				pos = 3;
			else if (!strncmp(textype, "Texture2DArray<", strlen("Texture2DArray<")))
				pos = 4; // float3 .xyz
			else if (!strncmp(textype, "Texture3D<", strlen("Texture3D<")))
				pos = 4;
			else if (!strncmp(textype, "TextureCube<", strlen("TextureCube<")))
				pos = 4;
			else if (!strncmp(textype, "TextureCubeArray<", strlen("TextureCubeArray<")))
				pos = 5; // float4 .xyzw
			else if (!strncmp(textype, "RWTexture1D<", strlen("RWTexture1D<")))
				pos = 2;
			else if (!strncmp(textype, "RWTexture1DArray<", strlen("RWTexture1DArray<")))
				pos = 3;
			else if (!strncmp(textype, "RWTexture2D<", strlen("RWTexture2D<")))
				pos = 3;
			else if (!strncmp(textype, "RWTexture2DArray<", strlen("RWTexture2DArray<")))
				pos = 4;
			else if (!strncmp(textype, "RWTexture3D<", strlen("RWTexture3D<")))
				pos = 4;
			else if (textype[0] == 0)
				pos = 5;
			else
				logDecompileError("  unknown texture type for truncation: " + string(textype));
			cpos = strrchr(op, '.');
			cpos[pos] = 0;
		}
	}

	// This routine was expecting only r0.xyz type input parameters, but we can also get float4(0,0,0,0) type
	// inputs as constants to things like .Load(). If it's a constant of any form, leave it unchanged.
	// The reason there is a second version of this .xyzw truncator, is because this version is for the _LD
	// operands, and for that version it's one parameter larger. Texture2D is 3 components, not 2 for this one.
	// The extra bracket for comparison is to avoid early mismatch like Texture1D instead of Texture1DArray.
	void truncateTextureLoadPos(char *op, const char *textype) {
		if (!strncmp(op, "float", 5))
			return;
		int pos = 5;
		if (!strncmp(textype, "Buffer<", strlen("Buffer<")))
			pos = 2; // int .x
		else if (!strncmp(textype, "Texture1D<", strlen("Texture1D<")))
			pos = 3; // int2 .xy
		else if (!strncmp(textype, "Texture2DMS<", strlen("Texture2DMS<")))
			pos = 3;
		else if (!strncmp(textype, "Texture1DArray<", strlen("Texture1DArray<")))
			pos = 4; // int3 .xyz
		else if (!strncmp(textype, "Texture2D<", strlen("Texture2D<")))
			pos = 4;
		else if (!strncmp(textype, "Texture2DMSArray<", strlen("Texture2DMSArray<")))
			pos = 4;
		else if (!strncmp(textype, "Texture2DArray<", strlen("Texture2DArray<")))
			pos = 5; // int4 .xyzw
		else if (!strncmp(textype, "Texture3D<", strlen("Texture3D<")))
			pos = 5;
		else if (!strncmp(textype, "ByteAddressBuffer", strlen("ByteAddressBuffer")))
			pos = 2;
		else if (!strncmp(textype, "RWByteAddressBuffer", strlen("RWByteAddressBuffer")))
			pos = 2;
		else if (!strncmp(textype, "RWBuffer<", strlen("RWBuffer<")))
			pos = 2;
		else if (!strncmp(textype, "RWTexture1D<", strlen("RWTexture1D<")))
			pos = 3;
		else if (!strncmp(textype, "RWTexture1DArray<", strlen("RWTexture1DArray<")))
			pos = 4;
		else if (!strncmp(textype, "RWTexture2D<", strlen("RWTexture2D<")))
			pos = 4;
		else if (!strncmp(textype, "RWTexture2DArray<", strlen("RWTexture2DArray<")))
			pos = 5;
		else if (!strncmp(textype, "RWTexture3D<", strlen("RWTexture3D<")))
			pos = 5;
		else if (textype[0] == 0)
			pos = 5;
		else
			logDecompileError("  unknown texture type for truncation: " + string(textype));
		char *cpos = strrchr(op, '.');
		cpos[pos] = 0;
	}

	void remapTarget(char *target) {
		char *pos = strchr(target, ',');
		if (pos)
			*pos = 0;
	}

	char *writeTarget(char *target) {
		StringStringMap::iterator i = mRemappedOutputRegisters.find(target);
		if (i != mRemappedOutputRegisters.end())
			strcpy_s(target, opcodeSize, i->second.c_str());
		return target;
	}

	string ci(string input) {
		// Translation for Hull Shader hardware inputs into standard HLSL control point variables
		if (mShaderType.substr(0, 3) == "hs_") {
			size_t ocpid = input.find("vOutputControlPointID");
			if (ocpid != string::npos) {
				input.replace(ocpid, strlen("vOutputControlPointID"), "i");
			}
			size_t fiid = input.find("vForkInstanceId");
			if (fiid != string::npos) {
				input.replace(fiid, strlen("vForkInstanceId"), "0");
			}
			size_t prim = input.find("vPrim");
			if (prim != string::npos) {
				input.replace(prim, strlen("vPrim"), "PatchID");
			}
			size_t vpos = 0;
			while ((vpos = input.find("[r", vpos)) != string::npos) {
				size_t dotPos = input.find('.', vpos);
				size_t closeBracket = input.find(']', vpos);
				if (dotPos != string::npos && dotPos < closeBracket) {
					input.insert(vpos + 1, "(int)");
					vpos += 6;
				} else {
					vpos += 2;
				}
			}
		}

		// Fix for replicated swizzles on remapped inputs (e.g. v1.www -> (float3)w1, v1.wwww -> (float4)w1)
		for (const auto &mapping : mRemappedInputRegisters) {
			const char *pos = strstr(input.c_str(), mapping.first.c_str());
			if (pos) {
				string ret(input.c_str(), pos);
				const char *after = pos + strlen(mapping.first.c_str());

				char remappedComp = mapping.first.back(); // e.g. 'w'
				size_t extraCount = 0;
				while (after[extraCount] == remappedComp)
					extraCount++;

				if (extraCount > 0 && (after[extraCount] == 0 || after[extraCount] == ' ' || after[extraCount] == ',' ||
									   after[extraCount] == ')' || after[extraCount] == ';')) {
					size_t totalComps = 1 + extraCount;
					string baseVar = mapping.second;
					if (baseVar.length() >= 2 && baseVar[baseVar.length() - 2] == '.')
						baseVar = baseVar.substr(0, baseVar.length() - 2);

					char castBuf[64];
					sprintf_s(castBuf, sizeof(castBuf), "(float%zu)%s", totalComps, baseVar.c_str());
					return ret + castBuf + string(after + extraCount);
				}

				return ret + mapping.second + string(after);
			}
		}
		return input;
	}

	// Only necessary for opcode_AND when used as boolean tests.
	// Converts literals of the form: l(0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000) to l(1.0, 1.0, 1.0, 1.0)
	//
	// In the microsoft world, 10 year old C specs are not interesting, so no stdclib function exists here for float,
	// and the std::hexfloat i/o stream fails to handle 32 bits properly. It's quite sad how weak this stuff is.
	// This took quite a long time to figure out how to make it work properly. The easiest answer seemed to be
	// to use reinterpret_cast from uint to float, but that appears to be unnecessary, as the %x assigned directly
	// to a float seems to work properly (but is not documented.)
	//
	// 1/1/15: New variant of l(-1) as an int, but used after a boolean state, so it's a comparison, not a bitand.
	// This would generate a QNAN as float. Made the hex comparison more strict with 0x%x, and thus that will fail
	// if it's an int type, and allow for converting that variant.
	void convertHexToFloat(char *target) {
		char convert[opcodeSize];
		int count;
		float lit[4];
		int ilit[4];
		int printed;
		if (target[0] == 'l') {
			printed = sprintf_s(convert, sizeof(convert), "l(");
			count = sscanf_s(target, "l(%x,%x,%x,%x)", (unsigned int *)&lit[0], (unsigned int *)&lit[1], (unsigned int *)&lit[2],
							 (unsigned int *)&lit[3]);
			if (count != 0) {
				for (int i = 0; i < count; i++)
					printed += sprintf_s(&convert[printed], sizeof(convert) - printed, "%f,", lit[i]);
			} else {
				count = sscanf_s(target, "l(%i,%i,%i,%i)", &ilit[0], &ilit[1], &ilit[2], &ilit[3]);
				assert(count != 0);
				for (int i = 0; i < count; i++)
					printed += sprintf_s(&convert[printed], sizeof(convert) - printed, "%i,", ilit[i]);
			}
			convert[printed - 1] = ')';
			strcpy_s(target, opcodeSize, convert);
		}
	}

	char *_convertToIntegerType(char *target, bool isUnsigned, bool fixupCBs) {
		char buffer[opcodeSize];
		int isMinus = target[0] == '-' ? 1 : 0;
		const char *prefix = isUnsigned ? "uint" : "int";
		if (!strncmp(target + isMinus, "int", 3) || !strncmp(target + isMinus, "uint", 4))
			return target;
		if (!strncmp(target + isMinus, "float", 5)) {
			int size = 0;
			float f0, f1, f2, f3;
			sscanf_s(target + isMinus, "float%d(%f,%f,%f,%f)", &size, &f0, &f1, &f2, &f3);
			buffer[0] = 0;
			if (isMinus)
				strcpy_s(buffer, opcodeSize, "-");
			if (size == 2)
				sprintf_s(buffer + strlen(buffer), opcodeSize - strlen(buffer), "%s2(%d,%d)", prefix, (int)f0, (int)f1);
			else if (size == 3)
				sprintf_s(buffer + strlen(buffer), opcodeSize - strlen(buffer), "%s3(%d,%d,%d)", prefix, (int)f0, (int)f1, (int)f2);
			else if (size == 4)
				sprintf_s(buffer + strlen(buffer), opcodeSize - strlen(buffer), "%s4(%d,%d,%d,%d)", prefix, (int)f0, (int)f1, (int)f2,
						  (int)f3);
			strcpy_s(target, opcodeSize, buffer);
			return target;
		}
		// Fixup for constant buffer reads where the type may not be known due to missing headers:
		if (fixupCBs && !strncmp(target + isMinus, "cb", 2)) {
			_snprintf_s(buffer, opcodeSize, opcodeSize, isUnsigned ? "asuint(%s)" : "asint(%s)", target);
			strcpy_s(target, opcodeSize, buffer);
			return target;
		}
		char *pos = strrchr(target, '.');
		if (pos) {
			size_t size = strlen(pos + 1);
			if (size == 1)
				_snprintf_s(buffer, opcodeSize, opcodeSize, "(%s)%s", prefix, target);
			else
				_snprintf_s(buffer, opcodeSize, opcodeSize, "(%s%Id)%s", prefix, size, target);
			strcpy_s(target, opcodeSize, buffer);
		}
		return target;
	}

	char *convertToInt(char *target) { return _convertToIntegerType(target, false, true); }
	char *convertToUInt(char *target) { return _convertToIntegerType(target, true, true); }
	char *castToInt(char *target) { return _convertToIntegerType(target, false, false); }
	char *castToUInt(char *target) { return _convertToIntegerType(target, true, false); }

	// The boolean check routines had the problem that they only looked for the actual register
	// name, like r1, instead of including the component, like r1.x. This fails in some cases
	// because another component can legally be used in between, which would not clear the
	// boolean use case.
	// The fix here is to make the elements of the set include the swizzle components too.
	//
	// 12-5-15: New change to all users of addBoolean, is that they will call a small helper
	// routine to do 'cmp', in order to return -1 or 0, instead of the HLSL 1 or 0. This
	// is intended to fix the problems we see where the assembly is using the -1 numerically
	// and not as a boolean. The helper is now just a macro "#define cmp -" to negate.
	void addBoolean(const char *arg) {
		string op = (arg[0] == '-') ? arg + 1 : arg;
		size_t dotspot = op.find('.');
		if (dotspot == string::npos) {
			mBooleanRegisters.insert(op);
			return;
		}
		string reg = op.substr(0, dotspot);
		for (size_t i = dotspot + 1; i < op.length(); i++)
			mBooleanRegisters.insert(reg + '.' + op[i]);
	}

	bool isBoolean(const char *arg) {
		if (mBooleanRegisters.empty())
			return false;
		string op = (arg[0] == '-') ? arg + 1 : arg;
		size_t dotspot = op.find('.');
		if (dotspot == string::npos)
			return mBooleanRegisters.find(op) != mBooleanRegisters.end();
		string reg = op.substr(0, dotspot);
		for (size_t i = dotspot + 1; i < op.length(); i++)
			if (mBooleanRegisters.find(reg + '.' + op[i]) != mBooleanRegisters.end())
				return true;
		return false;
	}

	void removeBoolean(const char *arg) {
		if (mBooleanRegisters.empty())
			return;
		string op = arg[0] == '-' ? arg + 1 : arg;
		size_t dotspot = op.find('.');
		if (dotspot == string::npos) {
			mBooleanRegisters.erase(op);
			return;
		}
		string reg = op.substr(0, dotspot);
		for (size_t i = dotspot + 1; i < op.length(); i++)
			mBooleanRegisters.erase(reg + '.' + op[i]);
	}

	char *fixImm(char *op, Operand &o) {
		if (o.eType == OPERAND_TYPE_IMMEDIATE32) {
			float oldValue;
			sscanf_s(op, "l(%e", &oldValue);
			if (!strncmp(op, "l(1.#INF00", strlen("l(1.#INF00")) || abs(oldValue - o.afImmediates[0]) < 0.1) {
				if (o.iNumComponents == 4)
					sprintf_s(op, opcodeSize, "l(%.9g,%.9g,%.9g,%.9g)", o.afImmediates[0], o.afImmediates[1], o.afImmediates[2],
							  o.afImmediates[3]);
				else if (o.iNumComponents == 3)
					sprintf_s(op, opcodeSize, "l(%.9g,%.9g,%.9g)", o.afImmediates[0], o.afImmediates[1], o.afImmediates[2]);
				else if (o.iNumComponents == 2)
					sprintf_s(op, opcodeSize, "l(%.9g,%.9g)", o.afImmediates[0], o.afImmediates[1]);
				else if (o.iNumComponents == 1)
					sprintf_s(op, opcodeSize, "l(%.9g)", o.afImmediates[0]);
			}
		}
		return op;
	}

	void WritePatches() {
		bool stereoParamsWritten = false;
		const char *StereoDecl =
			"\n\n// Auto-fixed shader\nfloat4 stereo = StereoParams.Load(0);\nfloat separation = stereo.x;\nfloat convergence = stereo.y;";

		// Vertex shader patches
		if (!mShaderType.substr(0, 3).compare("vs_")) {
			bool isMono = false;
			bool screenToWorldMatrix1 = false, screenToWorldMatrix2 = false;
			string backProjectVector1, backProjectVector2;
			if (!BackProject_Vector1.empty())
				backProjectVector1 = BackProject_Vector1.substr(0, BackProject_Vector1.find_first_of(".,"));
			if (!BackProject_Vector2.empty())
				backProjectVector2 = BackProject_Vector2.substr(0, BackProject_Vector2.find_first_of(".,"));
			for (const auto &entry : mCBufferData) {
				if (!screenToWorldMatrix1 && entry.second.Name == backProjectVector1)
					screenToWorldMatrix1 = true;
				if (!screenToWorldMatrix2 && entry.second.Name == backProjectVector2)
					screenToWorldMatrix2 = true;
			}
			if (screenToWorldMatrix1 || screenToWorldMatrix2) {
				mOutput.push_back(0);
				char *screenToWorldPos;
				if (screenToWorldMatrix1) {
					backProjectVector1 += '.';
					screenToWorldPos = strstr(mOutput.data(), backProjectVector1.c_str());
				} else {
					backProjectVector2 += '.';
					screenToWorldPos = strstr(mOutput.data(), backProjectVector2.c_str());
				}
				char *viewProjectMatrix = strstr(mOutput.data(), "ViewProjectionMatrix.");
				if (screenToWorldPos) {
					// This is a deferred rendering vertex shader.
					isMono = true;
					// Add view direction out parameter.
					char *lastPos = 0;
					char *pos = mOutput.data();
					while (pos) {
						lastPos = pos;
						pos = strstr(pos + 12, " : TEXCOORD");
					}
					if (lastPos && lastPos != mOutput.data()) {
						pos = strchr(lastPos, '\n');
						const char *viewDirectionDecl = "\nout float3 viewDirection : TEXCOORD31,";
						mOutput.insert(mOutput.begin() + (pos - mOutput.data()), viewDirectionDecl,
									   viewDirectionDecl + strlen(viewDirectionDecl));
					}
					// Add view direction calculation.
					char buf[512];
					if (screenToWorldMatrix1)
						sprintf_s(buf, sizeof(buf), "  viewDirection = float3(%s);\n", BackProject_Vector1.c_str());
					else
						sprintf_s(buf, sizeof(buf), "  viewDirection = float3(%s);\n", BackProject_Vector2.c_str());
					mOutput.insert(mOutput.end() - 1, buf, buf + strlen(buf));
					mPatched = true;

					// If we have a projection, make mono.
					if (viewProjectMatrix) {
						vector<char>::iterator writePos = mOutput.end() - 1;
						if (*writePos != '\n')
							--writePos;
						mOutput.insert(writePos, StereoDecl, StereoDecl + strlen(StereoDecl));
						stereoParamsWritten = true;
						char buffer[256];
						sprintf_s(buffer, sizeof(buffer), "%s.x -= separation * (%s.w - convergence);\n", mSV_Position.c_str(),
								  mSV_Position.c_str());
						mOutput.insert(mOutput.end() - 1, buffer, buffer + strlen(buffer));
					}
				}
				mOutput.pop_back();
			}

			// Process copies of SV_Position.
			if (!isMono && mFixSvPosition && mUsesProjection && !mSV_Position.empty()) {
				map<string, string>::iterator positionValue = mOutputRegisterValues.find(mSV_Position);
				if (positionValue != mOutputRegisterValues.end()) {
					size_t dotPos = positionValue->second.rfind('.');
					string rvalue = positionValue->second;
					if (dotPos > 0)
						rvalue = positionValue->second.substr(0, dotPos);
					for (map<string, string>::iterator i = mOutputRegisterValues.begin(); i != mOutputRegisterValues.end(); ++i) {
						if (!i->first.compare(mSV_Position))
							continue;
						map<string, DataType>::iterator dataType = mOutputRegisterType.find(i->first);
						if (dataType == mOutputRegisterType.end() || dataType->second != DT_float4)
							continue;
						dotPos = i->second.rfind('.');
						string rvalue2 = i->second;
						if (dotPos > 0)
							rvalue2 = i->second.substr(0, dotPos);
						if (!rvalue2.compare(rvalue)) {
							// Write params before return;.
							if (!stereoParamsWritten) {
								vector<char>::iterator writePos = mOutput.end() - 1;
								while (*writePos != '\n')
									--writePos;
								--writePos;
								while (*writePos != '\n')
									--writePos;
								mOutput.insert(writePos, StereoDecl, StereoDecl + strlen(StereoDecl));
								stereoParamsWritten = true;
							}
							vector<char>::iterator writePos = mOutput.end() - 1;
							while (*writePos != '\n')
								--writePos;
							--writePos;
							while (*writePos != '\n')
								--writePos;
							char buffer[256];
							string outputReg = i->first;
							if (outputReg.find('.') != string::npos)
								outputReg = outputReg.substr(0, outputReg.find('.'));
							sprintf_s(buffer, sizeof(buffer), "\n%s.x += separation * (%s.w - convergence);", outputReg.c_str(),
									  outputReg.c_str());
							mOutput.insert(writePos, buffer, buffer + strlen(buffer));
							mPatched = true;
						}
					}
				}
			}
			if (mRecompileVs)
				mPatched = true;
		}

		// Pixel shader patches
		if (!mShaderType.substr(0, 3).compare("ps_")) {
			// Search for depth texture.
			bool wposAvailable = false;
			map<int, string>::iterator depthTexture;
			for (depthTexture = mTextureNames.begin(); depthTexture != mTextureNames.end(); ++depthTexture) {
				if (depthTexture->second == ZRepair_DepthTexture1)
					break;
			}
			if (depthTexture != mTextureNames.end()) {
				long found = 0;
				for (const auto &entry : mCBufferData)
					for (unsigned int j = 0; j < ZRepair_Dependencies1.size(); ++j)
						if (entry.second.Name == ZRepair_Dependencies1[j])
							found |= 1 << j;
				if (!ZRepair_Dependencies1.size() || found == (1 << ZRepair_Dependencies1.size()) - 1) {
					mOutput.push_back(0);
					sprintf_s(op1, sizeof(op1), " = %s.Sample", ZRepair_DepthTexture1.c_str());
					char *pos = strstr(mOutput.data(), op1);
					ptrdiff_t searchPos = 0;
					while (pos) {
						char *bpos = pos;
						while (*--bpos != ' ')
							;
						string regName(bpos + 1, pos);
						char *endPos = strchr(pos, ',') + 2;
						bool constantDeclaration = endPos[0] == 'v' && endPos[1] >= '0' && endPos[1] <= '9';
						endPos = strchr(endPos, '\n');
						while (*--endPos != ')')
							;
						++endPos;
						pos += 3;
						string depthBufferStatement(pos, endPos);
						searchPos = endPos - mOutput.data();
						char buf[512];
						if (!wposAvailable)
							sprintf_s(buf, sizeof(buf),
									  "float4 zpos4 = %s;\nfloat zTex = zpos4.%c;\nfloat zpos = %s;\nfloat wpos = 1.0 / zpos;\n",
									  depthBufferStatement.c_str(), ZRepair_DepthTextureReg1, ZRepair_ZPosCalc1.c_str());
						else
							sprintf_s(buf, sizeof(buf), "zpos4 = %s;\nzTex = zpos4.%c;\nzpos = %s;\nwpos = 1.0 / zpos;\n",
									  depthBufferStatement.c_str(), ZRepair_DepthTextureReg1, ZRepair_ZPosCalc1.c_str());

						// There are a whole series of fixes to the statements that are doing mOutput.insert() - mOutput.begin();
						// We would get a series of Asserts (vector mismatch), but only rarely, usually at starting a new game in AC3.
						// The problem appears to be that mOutput.begin() was being calculated BEFORE the mOutput.insert(), which seems
						// to be the compiler optimizing. The problem seems to be that the Insert could move the mOutput array altogether,
						// and because of the use of vector<char>, the vector winds up being closest to a char* pointer.
						// With the pointer moved, the saved off mOutput.begin() was invalid and would cause the debug Assert.
						// By calculating that AFTER the insert has finished, we can avoid this rare buffer movement.
						// It took two days to figure out, it's worth some comments.
						if (constantDeclaration && !wposAvailable) {
							vector<char>::iterator iter = mOutput.insert(mOutput.begin() + mCodeStartPos, buf, buf + strlen(buf));
							mCodeStartPos = iter - mOutput.begin();
							mCodeStartPos += strlen(buf);
						} else if (!wposAvailable) {
							while (*pos != '\n')
								--pos;
							vector<char>::iterator iter =
								mOutput.insert(mOutput.begin() + (pos + 1 - mOutput.data()), buf, buf + strlen(buf));
							mCodeStartPos = iter - mOutput.begin();
							mCodeStartPos += strlen(buf);
						} else {
							while (*pos != '\n')
								--pos;
							mOutput.insert(mOutput.begin() + (pos + 1 - mOutput.data()), buf, buf + strlen(buf));
						}
						searchPos += strlen(buf);
						wposAvailable = true;
						pos = strstr(mOutput.data() + searchPos, op1);
					}
					mOutput.pop_back();
				}
			}
			if (!wposAvailable) {
				for (depthTexture = mTextureNames.begin(); depthTexture != mTextureNames.end(); ++depthTexture) {
					if (depthTexture->second == ZRepair_DepthTexture2)
						break;
				}
				if (depthTexture != mTextureNames.end()) {
					long found = 0;
					for (const auto &entry : mCBufferData)
						for (unsigned int j = 0; j < ZRepair_Dependencies2.size(); ++j)
							if (entry.second.Name == ZRepair_Dependencies2[j])
								found |= 1 << j;
					if (!ZRepair_Dependencies2.size() || found == (1 << ZRepair_Dependencies2.size()) - 1) {
						mOutput.push_back(0);
						sprintf_s(op1, sizeof(op1), " = %s.Sample", ZRepair_DepthTexture2.c_str());
						char *pos = strstr(mOutput.data(), op1);
						if (pos) {
							char *bpos = pos;
							while (*--bpos != ' ')
								;
							string regName(bpos + 1, pos);
							char *endPos = strchr(pos, ',') + 2;
							bool constantDeclaration = endPos[0] == 'v' && endPos[1] >= '0' && endPos[1] <= '9';
							endPos = strchr(endPos, '\n');
							while (*--endPos != ')')
								;
							++endPos;
							pos += 3;
							string depthBufferStatement(pos, endPos);
							vector<char>::iterator wpos = mOutput.begin();
							wpos = mOutput.erase(wpos + (pos - mOutput.data()), wpos + (endPos - mOutput.data()));
							const char ZPOS_REG[] = "zpos4";
							wpos = mOutput.insert(wpos, ZPOS_REG, ZPOS_REG + strlen(ZPOS_REG));
							char buf[256];
							sprintf_s(buf, sizeof(buf),
									  "float4 zpos4 = %s;\nfloat zTex = zpos4.%c;\nfloat zpos = %s;\nfloat wpos = 1.0 / zpos;\n",
									  depthBufferStatement.c_str(), ZRepair_DepthTextureReg2, ZRepair_ZPosCalc2.c_str());
							if (constantDeclaration) {
								vector<char>::iterator iter = mOutput.insert(mOutput.begin() + mCodeStartPos, buf, buf + strlen(buf));
								mCodeStartPos = iter - mOutput.begin();
								mCodeStartPos += strlen(buf);
							} else {
								while (*wpos != '\n')
									--wpos;
								vector<char>::iterator iter = mOutput.insert(wpos + 1, buf, buf + strlen(buf));
								mCodeStartPos = iter - mOutput.begin();
								mCodeStartPos += strlen(buf);
							}
							wposAvailable = true;
						}
						mOutput.pop_back();
					}
				}
			}

			// Search for position texture.
			if (!wposAvailable) {
				map<int, string>::iterator positionTexture;
				for (positionTexture = mTextureNames.begin(); positionTexture != mTextureNames.end(); ++positionTexture) {
					if (positionTexture->second == ZRepair_PositionTexture)
						break;
				}
				if (positionTexture != mTextureNames.end()) {
					mOutput.push_back(0);
					sprintf_s(op1, sizeof(op1), " = %s.Sample", ZRepair_PositionTexture.c_str());
					char *pos = strstr(mOutput.data(), op1);
					if (pos) {
						char *bpos = pos;
						while (*--bpos != ' ')
							;
						char buf[512];
						memcpy(buf, bpos + 1, pos - (bpos + 1));
						buf[pos - (bpos + 1)] = 0;
						applySwizzle(".xyz", buf);
						char calcStatement[256];
						sprintf_s(calcStatement, sizeof(calcStatement), ZRepair_WorldPosCalc.c_str(), buf);
						sprintf_s(buf, sizeof(buf), "\nfloat3 worldPos = %s;\nfloat zpos = worldPos.z;\nfloat wpos = 1.0 / zpos;",
								  calcStatement);
						pos = strchr(pos, '\n');
						vector<char>::iterator iter = mOutput.insert(mOutput.begin() + (pos - mOutput.data()), buf, buf + strlen(buf));
						mCodeStartPos = iter - mOutput.begin();
						mCodeStartPos += strlen(buf);
						wposAvailable = true;
					}
					mOutput.pop_back();
				}
			}

			// Add depth texture as a last resort, but only if it's specified in d3dx.ini
			// This is now inserted as the first parameter, because any later can generate
			// an error in some shaders complaining that user SV vars must come before system SV vars.
			// It seems to be fine, if not exactly correct, to have it as first parameter.
			if (!wposAvailable && mZRepair_DepthBuffer) {
				const char *INJECT_HEADER = "float4 zpos4 = InjectedDepthTexture.Load((int3) injectedScreenPos.xyz);\nfloat zpos = zpos4.x "
											"- 1;\nfloat wpos = 1.0 / zpos;\n";
				vector<char>::iterator iter =
					mOutput.insert(mOutput.begin() + mCodeStartPos, INJECT_HEADER, INJECT_HEADER + strlen(INJECT_HEADER));
				mCodeStartPos = iter - mOutput.begin();
				mCodeStartPos += strlen(INJECT_HEADER);
				char *pos = strstr(mOutput.data(), "void main(");
				while (*++pos != '\n')
					;
				assert(pos != NULL);
				const char *PARAM_HEADER = "\nfloat4 injectedScreenPos : SV_Position,";
				mOutput.insert(mOutput.begin() + (pos - mOutput.data()), PARAM_HEADER, PARAM_HEADER + strlen(PARAM_HEADER));
				mCodeStartPos += strlen(PARAM_HEADER);
				wposAvailable = true;
			}

			if (wposAvailable && InvTransforms.size()) {
				CBufferData::iterator keyFind;
				for (keyFind = mCBufferData.begin(); keyFind != mCBufferData.end(); ++keyFind) {
					bool found = false;
					for (vector<string>::iterator j = InvTransforms.begin(); j != InvTransforms.end(); ++j)
						if (keyFind->second.Name == *j)
							found = true;
					if (found)
						break;
					if (!ObjectPos_ID1.empty() && keyFind->second.Name.find(ObjectPos_ID1) != string::npos)
						break;
					if (!ObjectPos_ID2.empty() && keyFind->second.Name.find(ObjectPos_ID2) != string::npos)
						break;
					if (!MatrixPos_ID1.empty() && keyFind->second.Name == MatrixPos_ID1)
						break;
				}
				if (keyFind != mCBufferData.end()) {
					mOutput.push_back(0);
					if (!stereoParamsWritten) {
						if (mOutput[mCodeStartPos] != '\n')
							--mCodeStartPos;
						vector<char>::iterator iter =
							mOutput.insert(mOutput.begin() + mCodeStartPos, StereoDecl, StereoDecl + strlen(StereoDecl));
						mCodeStartPos = iter - mOutput.begin();
						mCodeStartPos += strlen(StereoDecl);
						stereoParamsWritten = true;
					}

					for (vector<string>::iterator invT = InvTransforms.begin(); invT != InvTransforms.end(); ++invT) {
						char buf[128];
						sprintf_s(buf, sizeof(buf), " %s._m00", invT->c_str());
						char *pos = strstr(mOutput.data(), buf);
						if (pos) {
							pos += strlen(buf);
							char *mpos = pos;
							while (*mpos != '*' && *mpos != '\n')
								++mpos;
							if (*mpos == '*') {
								char *bpos = mpos + 2;
								while (*bpos != ' ' && *bpos != ';')
									++bpos;
								string regName(mpos + 2, bpos);
								size_t dotPos = regName.rfind('.');
								if (dotPos != string::npos)
									regName = regName.substr(0, dotPos + 2);
								while (*mpos != '\n')
									--mpos;
								sprintf_s(buf, sizeof(buf), "\n%s -= separation * (wpos - convergence);", regName.c_str());
								mOutput.insert(mOutput.begin() + (mpos - mOutput.data()), buf, buf + strlen(buf));
								mPatched = true;
							} else {
								mpos = pos;
								while (*mpos != '(' && *mpos != '\n')
									--mpos;
								if (!strncmp(mpos - 3, "dot(", 4)) {
									char *bpos = mpos + 1;
									while (*bpos != ' ' && *bpos != ',' && *bpos != ';')
										++bpos;
									string regName(mpos + 1, bpos);
									size_t dotPos = regName.rfind('.');
									if (dotPos != string::npos)
										regName = regName.substr(0, dotPos + 2);
									while (*mpos != '\n')
										--mpos;
									sprintf_s(buf, sizeof(buf), "\n%s -= separation * (wpos - convergence);", regName.c_str());
									mOutput.insert(mOutput.begin() + (mpos - mOutput.data()), buf, buf + strlen(buf));
									mPatched = true;
								}
							}
						}
					}
					bool parameterWritten = false;
					const char *ParamPos1 = "SV_Position0,";
					const char *ParamPos2 = "\n  out ";
					const char *NewParam = "\nfloat3 viewDirection : TEXCOORD31,";

					if (!ObjectPos_ID1.empty()) {
						size_t offset = strstr(mOutput.data(), "void main(") - mOutput.data();
						while (offset < mOutput.size()) {
							char *pos = strstr(mOutput.data() + offset, ObjectPos_ID1.c_str());
							if (!pos)
								break;
							pos += ObjectPos_ID1.length();
							offset = pos - mOutput.data();
							if (*pos == '[')
								pos = strchr(pos, ']') + 1;
							if ((pos[1] == 'x' || pos[1] == 'y' || pos[1] == 'z') && (pos[2] == 'x' || pos[2] == 'y' || pos[2] == 'z') &&
								(pos[3] == 'x' || pos[3] == 'y' || pos[3] == 'z')) {
								char *lightPosDecl = pos + 1;
								while (*--pos != '=')
									;
								char *bpos = pos;
								while (*--bpos != '\n')
									;
								size_t size = (pos - 1) - (bpos + 3);
								memcpy(op1, bpos + 3, size);
								op1[size] = 0;
								strcpy_s(op2, op1);
								applySwizzle(".x", op2);
								strcpy_s(op3, op1);
								applySwizzle(".y", op3);
								strcpy_s(op4, op1);
								applySwizzle(".z", op4);
								char buf[512];
								if (ObjectPos_MUL1.empty())
									ObjectPos_MUL1 = string("1,1,1");
								sprintf_s(
									buf, sizeof(buf),
									"\nfloat3 stereoPos%dMul = float3(%s);\n%s += viewDirection.%c * separation * (wpos - convergence) * "
									"stereoPos%dMul.x;\n%s += viewDirection.%c * separation * (wpos - convergence) * stereoPos%dMul.y;\n%s "
									"+= viewDirection.%c * separation * (wpos - convergence) * stereoPos%dMul.z;",
									uuidVar, ObjectPos_MUL1.c_str(), op2, lightPosDecl[0], uuidVar, op3, lightPosDecl[1], uuidVar, op4,
									lightPosDecl[2], uuidVar);
								++uuidVar;
								pos = strchr(pos, '\n');
								mOutput.insert(mOutput.begin() + (pos - mOutput.data()), buf, buf + strlen(buf));
								offset += strlen(buf);
								mPatched = true;
								if (!parameterWritten) {
									char *posParam1 = strstr(mOutput.data(), ParamPos1);
									char *posParam2 = strstr(mOutput.data(), ParamPos2);
									char *posParam = posParam1 ? posParam1 : posParam2;
									while (*posParam != '\n')
										--posParam;
									mOutput.insert(mOutput.begin() + (posParam - mOutput.data()), NewParam, NewParam + strlen(NewParam));
									offset += strlen(NewParam);
									parameterWritten = true;
								}
							}
						}
					}

					if (!ObjectPos_ID2.empty()) {
						size_t offset = strstr(mOutput.data(), "void main(") - mOutput.data();
						while (offset < mOutput.size()) {
							char *pos = strstr(mOutput.data() + offset, ObjectPos_ID2.c_str());
							if (!pos)
								break;
							pos += ObjectPos_ID2.length();
							offset = pos - mOutput.data();
							if (*pos == '[')
								pos = strchr(pos, ']') + 1;
							if ((pos[1] == 'x' || pos[1] == 'y' || pos[1] == 'z') && (pos[2] == 'x' || pos[2] == 'y' || pos[2] == 'z') &&
								(pos[3] == 'x' || pos[3] == 'y' || pos[3] == 'z')) {
								char *spotPosDecl = pos + 1;
								while (*--pos != '=')
									;
								char *bpos = pos;
								while (*--bpos != '\n')
									;
								size_t size = (pos - 1) - (bpos + 3);
								memcpy(op1, bpos + 3, size);
								op1[size] = 0;
								strcpy_s(op2, op1);
								applySwizzle(".x", op2);
								strcpy_s(op3, op1);
								applySwizzle(".y", op3);
								strcpy_s(op4, op1);
								applySwizzle(".z", op4);
								char buf[512];
								if (ObjectPos_MUL2.empty())
									ObjectPos_MUL2 = string("1,1,1");
								sprintf_s(
									buf, sizeof(buf),
									"\nfloat3 stereoPos%dMul = float3(%s);\n%s += viewDirection.%c * separation * (wpos - convergence) * "
									"stereoPos%dMul.x;\n%s += viewDirection.%c * separation * (wpos - convergence) * stereoPos%dMul.y;\n%s "
									"+= viewDirection.%c * separation * (wpos - convergence) * stereoPos%dMul.z;",
									uuidVar, ObjectPos_MUL2.c_str(), op2, spotPosDecl[0], uuidVar, op3, spotPosDecl[1], uuidVar, op4,
									spotPosDecl[2], uuidVar);
								++uuidVar;
								pos = strchr(pos, '\n');
								mOutput.insert(mOutput.begin() + (pos - mOutput.data()), buf, buf + strlen(buf));
								offset += strlen(buf);
								mPatched = true;
								if (!parameterWritten) {
									char *posParam1 = strstr(mOutput.data(), ParamPos1);
									char *posParam2 = strstr(mOutput.data(), ParamPos2);
									char *posParam = posParam1 ? posParam1 : posParam2;
									while (*posParam != '\n')
										--posParam;
									mOutput.insert(mOutput.begin() + (posParam - mOutput.data()), NewParam, NewParam + strlen(NewParam));
									offset += strlen(NewParam);
									parameterWritten = true;
								}
							}
						}
					}

					if (!MatrixPos_ID1.empty()) {
						string ShadowPos1 = MatrixPos_ID1 + "._m00_m10_m20_m30 * ";
						string ShadowPos2 = MatrixPos_ID1 + "._m01_m11_m21_m31 * ";
						string ShadowPos3 = MatrixPos_ID1 + "._m02_m12_m22_m32 * ";
						char *pos1 = strstr(mOutput.data(), ShadowPos1.c_str());
						char *pos2 = strstr(mOutput.data(), ShadowPos2.c_str());
						char *pos3 = strstr(mOutput.data(), ShadowPos3.c_str());
						if (pos1 && pos2 && pos3) {
							string regName1(pos1 + ShadowPos1.length(), strchr(pos1 + ShadowPos1.length(), '.') + 2);
							string regName2(pos2 + ShadowPos2.length(), strchr(pos2 + ShadowPos2.length(), '.') + 2);
							string regName3(pos3 + ShadowPos3.length(), strchr(pos3 + ShadowPos3.length(), '.') + 2);
							char *pos = std::min(std::min(pos1, pos2), pos3);
							while (*--pos != '\n')
								;
							char buf[512];
							if (MatrixPos_MUL1.empty())
								MatrixPos_MUL1 = string("1,1,1");
							_snprintf_s(buf, 512, 512,
										"\nfloat3 stereoMat%dMul = float3(%s);\n%s -= viewDirection.x * separation * (wpos - convergence) "
										"* stereoMat%dMul.x;\n%s -= viewDirection.y * separation * (wpos - convergence) * "
										"stereoMat%dMul.y;\n%s -= viewDirection.z * separation * (wpos - convergence) * stereoMat%dMul.z;",
										uuidVar, MatrixPos_MUL1.c_str(), regName1.c_str(), uuidVar, regName2.c_str(), uuidVar,
										regName3.c_str(), uuidVar);
							++uuidVar;
							mOutput.insert(mOutput.begin() + (pos - mOutput.data()), buf, buf + strlen(buf));
							mPatched = true;
							if (!parameterWritten) {
								char *posParam1 = strstr(mOutput.data(), ParamPos1);
								char *posParam2 = strstr(mOutput.data(), ParamPos2);
								char *posParam = posParam1 ? posParam1 : posParam2;
								if (posParam != NULL) {
									while (*posParam != '\n')
										--posParam;
									mOutput.insert(mOutput.begin() + (posParam - mOutput.data()), NewParam, NewParam + strlen(NewParam));
									parameterWritten = true;
								}
							}
						}
					}
					mOutput.pop_back();
				}
			}
		}
	}

	// Common routine for handling the no header info case for declarations.
	// This is grim, because we add the '4' to the format by default, as we
	// cannot determine the actual size of a declaration without reflection data.
	// It will mostly work, but generate hand-fix errors at times.
	void CreateRawFormat(const string &texType, int bufIndex) {
		char buffer[128];
		char format[16];
		sprintf_s(buffer, sizeof(buffer), "t%d", bufIndex);
		mTextureNames[bufIndex] = buffer;
		sscanf_s(op1, "(%[^,]", format, 16);
		string form4 = string(format) + "4";
		mTextureType[bufIndex] = texType + "<" + form4 + ">";
		sprintf_s(buffer, sizeof(buffer), "%s t%d : register(t%d);\n\n", mTextureType[bufIndex].c_str(), bufIndex, bufIndex);
		mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
		mCodeStartPos += strlen(buffer);
	}

	// General output routine while decoding shader, to automatically apply indenting to HLSL code.
	// When we see a '{' or '}' we'll increase or decrease the indent.
	void appendOutput(const char *line) {
		char splitLine[1024];
		const char *effectiveLine = line;
		const char *eq = strchr(line, '=');

		// Universal fix: split ANY instruction writing a vector into split output registers (e.g. o5.xyzw, o2.zw, etc.)
		if (eq && mShaderType.substr(0, 3) != "hs_") {
			int slot = -1;
			char mask[16];
			const char *oPos = line;
			while (oPos < eq) {
				oPos = strchr(oPos, 'o');
				if (!oPos || oPos >= eq)
					break;
				if ((oPos == line || *(oPos - 1) == ' ' || *(oPos - 1) == '\t' || *(oPos - 1) == '.') &&
					sscanf_s(oPos, "o%d.%[xyzws]", &slot, mask, (int)sizeof(mask)) == 2) {
					string baseO = "o" + to_string(slot);
					if (mOutputRegisterType.find(baseO) != mOutputRegisterType.end()) {
						bool needsSplit = false;
						for (size_t k = 0; k < strlen(mask); ++k) {
							string singleComp = baseO + ".";
							singleComp.push_back(mask[k]);
							if (mRemappedOutputRegisters.find(singleComp) != mRemappedOutputRegisters.end()) {
								needsSplit = true;
								break;
							}
						}
						string fullReg = baseO + "." + string(mask);
						if (needsSplit && mRemappedOutputRegisters.find(fullReg) == mRemappedOutputRegisters.end()) {
							string expr = eq + 1;
							while (!expr.empty() &&
								   (expr.back() == '\n' || expr.back() == '\r' || expr.back() == ' ' || expr.back() == ';'))
								expr.pop_back();
							size_t firstNonSpace = expr.find_first_not_of(" \t");
							if (firstNonSpace != string::npos)
								expr = expr.substr(firstNonSpace);
							string newLines = "";
							char fdestAssign[512];
							sprintf_s(fdestAssign, sizeof(fdestAssign), "  fDest.%s = %s;\n", mask, expr.c_str());
							newLines += fdestAssign;
							for (size_t k = 0; k < strlen(mask); ++k) {
								string singleComp = baseO + ".";
								singleComp.push_back(mask[k]);
								string target = singleComp;
								if (mRemappedOutputRegisters.find(singleComp) != mRemappedOutputRegisters.end()) {
									target = mRemappedOutputRegisters[singleComp];
								}
								char compAssign[256];
								sprintf_s(compAssign, sizeof(compAssign), "  %s = fDest.%c;\n", target.c_str(), mask[k]);
								newLines += compAssign;
							}
							strcpy_s(splitLine, sizeof(splitLine), newLines.c_str());
							effectiveLine = splitLine;
							break;
						}
					}
				}
				oPos++;
			}
		}

		bool open = (strchr(effectiveLine, '{') != NULL);
		bool close = (strchr(effectiveLine, '}') != NULL);
		if (close)
			nestCount--;

		// Route output to appropriate buffer based on current Hull Shader phase
		vector<char> *targetBuffer = &mOutput;
		if (mShaderType.substr(0, 3) == "hs_") {
			if (mHSPhase == 1)
				targetBuffer = &mHSControlPointPhase;
			else if (mHSPhase == 2)
				targetBuffer = &mHSForkPhase;
			else if (mHSPhase == 0)
				return;
		}

		for (int i = 0; i < nestCount; i++)
			targetBuffer->insert(targetBuffer->end(), indent, indent + strlen(indent));
		if (open)
			nestCount++;
		targetBuffer->insert(targetBuffer->end(), effectiveLine, effectiveLine + strlen(effectiveLine));
	}

	// =========================================================================
	// Instruction Block Code Parser
	// =========================================================================
	void ParseCode(Shader *shader, const char *c, size_t size) {
		mOutputRegisterValues.clear();
		mBooleanRegisters.clear();
		mCodeStartPos = mOutput.size();

		char buffer[2048];
		size_t pos = 0;
		unsigned int iNr = 0;
		bool skip_shader = false;

		// Deduplicated helper lambdas for opcode output formatting
		auto emitUnaryMath = [&](const char *funcName, bool handleBool = true) {
			remapTarget(op1);
			applySwizzle(op1, op2);
			if (!shader->psInst[iNr].bSaturate)
				sprintf_s(buffer, sizeof(buffer), "  %s = %s(%s);\n", writeTarget(op1), funcName, ci(op2).c_str());
			else
				sprintf_s(buffer, sizeof(buffer), "  %s = saturate(%s(%s));\n", writeTarget(op1), funcName, ci(op2).c_str());
			appendOutput(buffer);
			if (handleBool)
				removeBoolean(op1);
		};

		auto emitComparison = [&](Instruction *inst, const char *cmpOp, bool isInt = false, bool isUInt = false) {
			remapTarget(op1);
			if (isInt) {
				applySwizzle(op1, op2, true);
				applySwizzle(op1, op3, true);
				sprintf_s(buffer, sizeof(buffer), "  %s = cmp(%s %s %s);\n", writeTarget(op1), ci(convertToInt(op2)).c_str(), cmpOp,
						  ci(convertToInt(op3)).c_str());
			} else if (isUInt) {
				applySwizzle(op1, op2, true);
				applySwizzle(op1, op3, true);
				sprintf_s(buffer, sizeof(buffer), "  %s = cmp(%s %s %s);\n", writeTarget(op1), ci(convertToUInt(op2)).c_str(), cmpOp,
						  ci(convertToUInt(op3)).c_str());
			} else {
				applySwizzle(op1, fixImm(op2, inst->asOperands[1]));
				applySwizzle(op1, fixImm(op3, inst->asOperands[2]));
				sprintf_s(buffer, sizeof(buffer), "  %s = cmp(%s %s %s);\n", writeTarget(op1), ci(op2).c_str(), cmpOp, ci(op3).c_str());
			}
			appendOutput(buffer);
			addBoolean(op1);
		};

		auto emitDot = [&](Instruction *inst, const char *swiz) {
			remapTarget(op1);
			applySwizzle(swiz, fixImm(op2, inst->asOperands[1]));
			applySwizzle(swiz, fixImm(op3, inst->asOperands[2]));
			if (!inst->bSaturate)
				sprintf_s(buffer, sizeof(buffer), "  %s = dot(%s, %s);\n", writeTarget(op1), ci(op2).c_str(), ci(op3).c_str());
			else
				sprintf_s(buffer, sizeof(buffer), "  %s = saturate(dot(%s, %s));\n", writeTarget(op1), ci(op2).c_str(), ci(op3).c_str());
			appendOutput(buffer);
			removeBoolean(op1);
		};

		auto emitDeriv = [&](const char *derivFunc) {
			remapTarget(op1);
			applySwizzle(op1, op2);
			sprintf_s(buffer, sizeof(buffer), "  %s = %s(%s);\n", writeTarget(op1), derivFunc, op2);
			appendOutput(buffer);
			removeBoolean(op1);
		};

		auto emitAtomicStub = [&](const char *sig) {
			sprintf_s(buffer, sizeof(buffer), "  // Needs manual fix for instruction:\n");
			appendOutput(buffer);
			ASMLineOut(c, pos, size);
			sprintf_s(buffer, sizeof(buffer), "  %s\n", sig);
			appendOutput(buffer);
		};

		while (pos < size && iNr < shader->psInst.size()) {
			Instruction *instr = &shader->psInst[iNr];

			// Now ignore '#line' or 'undecipherable' debug info (DefenseGrid2)
			if (!strncmp(c + pos, "#line", 5) || !strncmp(c + pos, "undecipherable", 14)) {
				NextLine(c, pos, size);
				continue;
			}
			// And the dissassembler apparently can add trailing \0 characters
			// and blank lines to the buffer in the debug case.
			if (c[pos] == 0x0a || c[pos] == 0x00 || c[pos] == 0x0d) {
				pos++;
				continue;
			}
			// Ignore comments
			if (!strncmp(c + pos, "//", 2)) {
				NextLine(c, pos, size);
				continue;
			}
			// Read statement
			if (ReadStatement(c + pos) < 1) {
				if (c[pos] == ' ' || c[pos] == '\t' || c[pos] == '\r') {
					NextLine(c, pos, size);
					continue;
				}
				logDecompileError("Error parsing statement: " + string(c + pos, 80));
				NextLine(c, pos, size);
				continue;
			}

			// Some shaders seen in World of Diving contain multiple shader programs.
			// Ignore any instructions from old shader models that we do not handle to
			// avoid crashes.
			if (!strncmp(statement, "vs_1", 4) || !strncmp(statement, "vs_2", 4) || !strncmp(statement, "ps_1", 4) ||
				!strncmp(statement, "ps_2", 4)) {
				skip_shader = true;
				NextLine(c, pos, size);
				continue;
			}
			if (!strncmp(statement, "vs_", 3) || !strncmp(statement, "hs_", 3) || !strncmp(statement, "ds_", 3) ||
				!strncmp(statement, "gs_", 3) || !strncmp(statement, "ps_", 3) || !strncmp(statement, "cs_", 3)) {
				skip_shader = false;
				mShaderType = statement;
			} else if (skip_shader) {
				NextLine(c, pos, size);
				continue;
			} else if (!strcmp(statement, "hs_control_point_phase")) {
				mHSPhase = 1;
				nestCount = 1;
				if (iNr < shader->psInst.size() && shader->psInst[iNr].eOpcode == OPCODE_HS_CONTROL_POINT_PHASE)
					iNr++;
			} else if (!strcmp(statement, "hs_fork_phase") || !strcmp(statement, "hs_join_phase")) {
				mHSPhase = 2;
				nestCount = 1;
				if (iNr < shader->psInst.size() &&
					(shader->psInst[iNr].eOpcode == OPCODE_HS_FORK_PHASE || shader->psInst[iNr].eOpcode == OPCODE_HS_JOIN_PHASE))
					iNr++;
			} else if (!strcmp(statement, "}")) {
				NextLine(c, pos, size);
				continue;
			} else if (!strcmp(statement, "hs_decls") || !strcmp(statement, "dcl_hs_max_tessfactor") ||
					   !strcmp(statement, "dcl_input_control_point_count") || !strcmp(statement, "dcl_output_control_point_count") ||
					   !strcmp(statement, "dcl_tessellator_partitioning") || !strcmp(statement, "dcl_tessellator_output_primitive") ||
					   !strcmp(statement, "dcl_tessellator_domain")) {
				// Handled in ParseCodeOnlyShaderType
			} else if (!strcmp(statement, "dcl_immediateConstantBuffer")) {
				sprintf_s(buffer, sizeof(buffer), "  const float4 icb[] =");
				appendOutput(buffer);
				pos += strlen(statement);
				while (c[pos] != 0x0a && pos < size) {
					char b[2] = {c[pos++], 0};
					appendOutput(b);
				}
				appendOutput("\n");
			} else if (!strcmp(statement, "dcl_constantbuffer")) {
				char *strPos = strstr(op1, "cb");
				if (!strPos)
					strPos = strstr(op1, "CB");
				if (strPos) {
					int bufIndex = 0;
					if (sscanf_s(strPos + 2, "%d", &bufIndex) != 1) {
						logDecompileError("Error parsing buffer register index: " + string(op1));
						NextLine(c, pos, size);
						continue;
					}
					strPos = strchr(op1, '[');
					if (!strPos) {
						logDecompileError("5 Error parsing buffer offset: " + string(op1));
						NextLine(c, pos, size);
						continue;
					}
					int bufSize = 0;
					if (sscanf_s(strPos + 1, "%d", &bufSize) != 1) {
						logDecompileError("Error parsing buffer size: " + string(op1));
						NextLine(c, pos, size);
						continue;
					}
					// In the case where headers have been stripped, we will not have any CBufferNames,
					// but we will have these dcl_constantbuffer declarations. We are now using
					// the absence of any mCBufferNames as indication that we have stripped headers,
					// and create the fake cb2 style names as the best we can do.
					// Not sure this will work in all cases, because the offsets into the buffer are
					// not required to be zero for the first element, but we have no other info here.
					if (mCBufferNames.empty()) {
						BufferEntry e;
						e.bt = DT_float4;
						e.matrixRow = 0;
						e.isRowMajor = false;
						sprintf_s(buffer, sizeof(buffer), "cbuffer cb%d : register(b%d)\n{\n  float4 cb%d[%d];\n}\n\n", bufIndex, bufIndex,
								  bufIndex, bufSize);
						vector<char>::iterator ipos = mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
						mCodeStartPos += strlen(buffer);
						ipos += strlen(buffer);
						for (int j = 0; j < bufSize; ++j) {
							sprintf_s(buffer, sizeof(buffer), "cb%d[%d]", bufIndex, j);
							e.Name = buffer;
							mCBufferData[(bufIndex << 16) + j * 16] = e;
						}
					}
				}
			} else if (!strcmp(statement, "dcl_sampler")) {
				if (op1[0] == 's' || op1[0] == 'S') {
					int bufIndex = parseResourceIndex(op1);
					if (!strcmp(op2, "mode_default")) {
						map<int, string>::iterator i = mSamplerNames.find(bufIndex);
						if (i == mSamplerNames.end()) {
							sprintf_s(buffer, sizeof(buffer), "s%d_s", bufIndex);
							mSamplerNames[bufIndex] = buffer;
							sprintf_s(buffer, sizeof(buffer), "SamplerState %s : register(s%d);\n\n", mSamplerNames[bufIndex].c_str(),
									  bufIndex);
							mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
							mCodeStartPos += strlen(buffer);
						}
					} else if (!strcmp(op2, "mode_comparison")) {
						map<int, string>::iterator i = mSamplerComparisonNames.find(bufIndex);
						if (i == mSamplerComparisonNames.end()) {
							sprintf_s(buffer, sizeof(buffer), "s%d_s", bufIndex);
							mSamplerComparisonNames[bufIndex] = buffer;
							sprintf_s(buffer, sizeof(buffer), "SamplerComparisonState %s : register(s%d);\n\n",
									  mSamplerComparisonNames[bufIndex].c_str(), bufIndex);
							mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
							mCodeStartPos += strlen(buffer);
						}
					} else {
						logDecompileError("Error parsing dcl_sampler type: " + string(op2));
						NextLine(c, pos, size);
						continue;
					}
				}
			}
			// For all of these dcl_resource_texture* variants, if we don't have any header information
			// available, we try to fall back to the raw text definitions, and use registers instead of
			// named variable. A big problem though is that the count is unknown if we have no headers,
			// so it should be Texture2D<float2> for example, but we only see (float,float,float,float).
			// With no headers and no reflection this can't be done, so for now we'll set them to <*4>
			// as the most common use case. But only for this case where the game is being hostile.
			// It will probably require hand tuning to correct usage of those texture registers.
			// Also unlikely to work for the (mixed,mixed,mixed,mixed) case. <mixed4> will be wrong.
			else if (!strcmp(statement, "dcl_resource_texture2d")) {
				if (op2[0] == 't' || op2[0] == 'T') {
					int bufIndex = 0;
					if (sscanf_s(op2 + 1, "%d", &bufIndex) != 1) {
						logDecompileError("Error parsing texture register index: " + string(op2));
						NextLine(c, pos, size);
						continue;
					}
					map<int, string>::iterator i = mTextureNames.find(bufIndex);
					if (i == mTextureNames.end())
						CreateRawFormat("Texture2D", bufIndex);
				}
			} else if (!strcmp(statement, "dcl_resource_texture2darray")) {
				if (op2[0] == 't' || op2[0] == 'T') {
					int bufIndex = 0;
					if (sscanf_s(op2 + 1, "%d", &bufIndex) != 1) {
						logDecompileError("Error parsing texture2darray register index: " + string(op2));
						NextLine(c, pos, size);
						continue;
					}
					map<int, string>::iterator i = mTextureNames.find(bufIndex);
					if (i == mTextureNames.end())
						CreateRawFormat("Texture2DArray", bufIndex);
				}
			} else if (!strncmp(statement, "dcl_resource_texture2dms", strlen("dcl_resource_texture2dms"))) {
				if (op2[0] == 't' || op2[0] == 'T') {
					int bufIndex = 0;
					if (sscanf_s(op2 + 1, "%d", &bufIndex) != 1) {
						logDecompileError("Error parsing dcl_resource_texture2dms register index: " + string(op2));
						NextLine(c, pos, size);
						continue;
					}
					int dim = 0;
					if (sscanf_s(statement, "dcl_resource_texture2dms(%d)", &dim) != 1) {
						logDecompileError("Error parsing dcl_resource_texture2dms array dimension: " + string(statement));
						NextLine(c, pos, size);
						continue;
					}
					map<int, string>::iterator i = mTextureNames.find(bufIndex);
					if (i == mTextureNames.end()) {
						sprintf_s(buffer, sizeof(buffer), "t%d", bufIndex);
						mTextureNames[bufIndex] = buffer;
						char format[16];
						sscanf_s(op1, "(%[^,]", format, 16);
						string form4 = string(format) + "4";
						mTextureType[bufIndex] = "Texture2DMS<" + form4 + ">";
						if (dim == 0)
							sprintf_s(buffer, sizeof(buffer), "Texture2DMS<%s> t%d : register(t%d);\n\n", form4.c_str(), bufIndex,
									  bufIndex);
						else
							sprintf_s(buffer, sizeof(buffer), "Texture2DMS<%s,%d> t%d : register(t%d);\n\n", form4.c_str(), dim, bufIndex,
									  bufIndex);
						mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
						mCodeStartPos += strlen(buffer);
					}
				}
			} else if (!strcmp(statement, "dcl_resource_texture3d")) {
				if (op2[0] == 't' || op2[0] == 'T') {
					int bufIndex = 0;
					if (sscanf_s(op2 + 1, "%d", &bufIndex) != 1) {
						logDecompileError("Error parsing texture register index: " + string(op2));
						NextLine(c, pos, size);
						continue;
					}
					// Create if not existing. e.g. if no ResourceBinding section in ASM.
					map<int, string>::iterator i = mTextureNames.find(bufIndex);
					if (i == mTextureNames.end())
						CreateRawFormat("Texture3D", bufIndex);
				}
			} else if (!strcmp(statement, "dcl_resource_texturecube")) {
				if (op2[0] == 't' || op2[0] == 'T') {
					int bufIndex = 0;
					if (sscanf_s(op2 + 1, "%d", &bufIndex) != 1) {
						logDecompileError("Error parsing texture register index: " + string(op2));
						NextLine(c, pos, size);
						continue;
					}
					// Create if not existing. e.g. if no ResourceBinding section in ASM.
					map<int, string>::iterator i = mTextureNames.find(bufIndex);
					if (i == mTextureNames.end())
						CreateRawFormat("TextureCube", bufIndex);
				}
			} else if (!strcmp(statement, "dcl_resource_texturecubearray")) {
				if (op2[0] == 't' || op2[0] == 'T') {
					int bufIndex = 0;
					if (sscanf_s(op2 + 1, "%d", &bufIndex) != 1) {
						logDecompileError("Error parsing texture register index: " + string(op2));
						NextLine(c, pos, size);
						continue;
					}
					// Create if not existing. e.g. if no ResourceBinding section in ASM.
					map<int, string>::iterator i = mTextureNames.find(bufIndex);
					if (i == mTextureNames.end())
						CreateRawFormat("TextureCubeArray", bufIndex);
				}
			} else if (!strcmp(statement, "dcl_resource_buffer")) // dcl_resource_buffer (sint,sint,sint,sint) t2
			{
				if (op2[0] == 't' || op2[0] == 'T') {
					int bufIndex = 0;
					if (sscanf_s(op2 + 1, "%d", &bufIndex) != 1) {
						logDecompileError("Error parsing texture register index: " + string(op2));
						NextLine(c, pos, size);
						continue;
					}
					// Create if not existing. e.g. if no ResourceBinding section in ASM.
					map<int, string>::iterator i = mTextureNames.find(bufIndex);
					if (i == mTextureNames.end())
						CreateRawFormat("Buffer", bufIndex);
				}
			}
			// SM5/5.1 structured and raw buffer resources. These use the texture (t#)
			// register space, and the resource name sits in op1 (e.g. "T0,") with the
			// element stride following. With reflection stripped we can only emit a
			// best-effort <float4> / ByteAddressBuffer declaration, but recording the
			// type here keeps the .Load() truncation logic from failing.
			else if (!strcmp(statement, "dcl_resource_structured")) // dcl_resource_structured T0, 16 space=0,reg=0
			{
				if (op1[0] == 't' || op1[0] == 'T') {
					int bufIndex = parseResourceIndex(op1);
					if (mTextureNames.find(bufIndex) == mTextureNames.end()) {
						sprintf_s(buffer, sizeof(buffer), "t%d", bufIndex);
						mTextureNames[bufIndex] = buffer;
						mTextureType[bufIndex] = "StructuredBuffer<float4>";
						sprintf_s(buffer, sizeof(buffer), "StructuredBuffer<float4> t%d : register(t%d);\n\n", bufIndex, bufIndex);
						mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
						mCodeStartPos += strlen(buffer);
					}
				}
			} else if (!strcmp(statement, "dcl_resource_raw")) // dcl_resource_raw T0 space=0,reg=0
			{
				if (op1[0] == 't' || op1[0] == 'T') {
					int bufIndex = parseResourceIndex(op1);
					if (mTextureNames.find(bufIndex) == mTextureNames.end()) {
						sprintf_s(buffer, sizeof(buffer), "t%d", bufIndex);
						mTextureNames[bufIndex] = buffer;
						mTextureType[bufIndex] = "ByteAddressBuffer";
						sprintf_s(buffer, sizeof(buffer), "ByteAddressBuffer t%d : register(t%d);\n\n", bufIndex, bufIndex);
						mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
						mCodeStartPos += strlen(buffer);
					}
				}
			}
			// Typed UAV resources (RWTexture / RWBuffer). These live in the u# register
			// space and the current opcode handlers for them are stubs, so we only emit a
			// reference declaration rather than registering them in the texture maps.
			else if (!strncmp(statement, "dcl_uav_typed_", strlen("dcl_uav_typed_"))) {
				const char *dim = statement + strlen("dcl_uav_typed_");
				const char *rwType = "RWTexture2D";
				if (!strcmp(dim, "buffer"))
					rwType = "RWBuffer";
				else if (!strcmp(dim, "texture1d"))
					rwType = "RWTexture1D";
				else if (!strcmp(dim, "texture1darray"))
					rwType = "RWTexture1DArray";
				else if (!strcmp(dim, "texture2d"))
					rwType = "RWTexture2D";
				else if (!strcmp(dim, "texture2darray"))
					rwType = "RWTexture2DArray";
				else if (!strcmp(dim, "texture3d"))
					rwType = "RWTexture3D";
				int uavIndex = parseResourceIndex(op2); // (float,float,float,float) U0 ...
				char format[16] = "float";
				sscanf_s(op1, "(%[^,]", format, 16);
				sprintf_s(buffer, sizeof(buffer), "%s<%s4> u%d : register(u%d);\n\n", rwType, format, uavIndex, uavIndex);
				mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
				mCodeStartPos += strlen(buffer);
			} else if (!strcmp(statement, "dcl_uav_structured")) // dcl_uav_structured U0, 16 space=0,reg=0
			{
				int uavIndex = parseResourceIndex(op1);
				sprintf_s(buffer, sizeof(buffer), "RWStructuredBuffer<float4> u%d : register(u%d);\n\n", uavIndex, uavIndex);
				mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
				mCodeStartPos += strlen(buffer);
			} else if (!strcmp(statement, "dcl_uav_raw")) // dcl_uav_raw U0 space=0,reg=0
			{
				int uavIndex = parseResourceIndex(op1);
				sprintf_s(buffer, sizeof(buffer), "RWByteAddressBuffer u%d : register(u%d);\n\n", uavIndex, uavIndex);
				mOutput.insert(mOutput.begin(), buffer, buffer + strlen(buffer));
				mCodeStartPos += strlen(buffer);
			} else if (!strcmp(statement, "{")) {
				// Declaration from
				// dcl_immediateConstantBuffer { { 1.000000, 0, 0, 0},
				//                      { 0, 1.000000, 0, 0},
				//                      { 0, 0, 1.000000, 0},
				//                      { 0, 0, 0, 1.000000} }
				while (c[pos] != 0x0a && pos < size) {
					char b[2] = {c[pos++], 0};
					appendOutput(b);
				}
				if (c[pos - 1] == '}')
					appendOutput(";");
				char b[2] = {c[pos++], 0};
				appendOutput(b);
				continue;
			} else if (!strcmp(statement, "dcl_indexrange")) {
				int numIndex = 0;
				sscanf_s(op2, "%d", &numIndex);
				sprintf_s(buffer, sizeof(buffer), "  float4 v[%d] = { ", numIndex);
				for (int i = 0; i < numIndex; ++i)
					sprintf_s(buffer + strlen(buffer), sizeof(buffer) - strlen(buffer), "v%d,", i);
				buffer[strlen(buffer) - 1] = 0;
				strcat_s(buffer, sizeof(buffer), " };\n");
				appendOutput(buffer);
			} else if (!strcmp(statement, "dcl_indexableTemp")) {
				// Always returned 4, not actual index. ToDo: likely not always float4.
				// format as: dcl_indexableTemp x0[40], 4
				int numIndex = 0;
				char varName[opcodeSize];
				sscanf_s(op1, "%[^[][%d]", varName, opcodeSize, &numIndex);
				sprintf_s(buffer, sizeof(buffer), "  float4 %s[%d];\n", varName, numIndex);
				appendOutput(buffer);
			} else if (!strcmp(statement, "dcl_input")) {
				// Can have 'vCoverage' variable implicitly defined,
				// not in input signature when reflection is stripped.
				if (!strcmp(op1, "vCoverage"))
					appendOutput("  uint vCoverage : SV_Coverage,\n");
			} else if (!strcmp(statement, "dcl_temps")) {
				int numTemps = 0;
				sscanf_s(c + pos, "%s %d", statement, (int)sizeof(statement), &numTemps);
				if (mShaderType.substr(0, 3) == "hs_") {
					vector<int> newTemps;
					set<int> &activeSet = (mHSPhase == 1) ? mDeclaredTempsHS_Main : mDeclaredTempsHS_Fork;
					for (int i = 0; i < numTemps; ++i) {
						if (activeSet.find(i) == activeSet.end()) {
							activeSet.insert(i);
							newTemps.push_back(i);
						}
					}
					if (!newTemps.empty()) {
						appendOutput("  float4 ");
						for (size_t i = 0; i < newTemps.size(); ++i) {
							sprintf_s(buffer, sizeof(buffer), "r%d%s", newTemps[i], (i + 1 == newTemps.size()) ? ";\n" : ", ");
							appendOutput(buffer);
						}
					}
				} else {
					if (numTemps > 0) {
						string tempDecl = "  float4 ";
						for (int i = 0; i < numTemps; ++i) {
							tempDecl += "r" + to_string(i) + (i + 1 == numTemps ? ";\n" : ", ");
						}
						appendOutput(tempDecl.c_str());
					}
				}
			}
			// For Geometry Shaders, e.g. dcl_stream m0 TODO: make it StreamN, add to varlist
			else if (!strcmp(statement, "dcl_stream")) {
				// Write out original ASM, inline, for reference.
				sprintf_s(buffer, sizeof(buffer), "// Needs manual fix for instruction:  \n//");
				appendOutput(buffer);
				ASMLineOut(c, pos, size);
				sprintf_s(buffer, sizeof(buffer), "  inout TriangleStream<float> m0,\n");
				appendOutput(buffer);
			}
			// For Geometry Shaders, e.g. dcl_maxout n
			else if (!strcmp(statement, "dcl_maxout")) {
				char *main_ptr = strstr(mOutput.data(), "void main(");
				size_t offset = main_ptr - mOutput.data();
				sprintf_s(buffer, sizeof(buffer), "[maxvertexcount(%s)]\n", op1);
				mOutput.insert(mOutput.begin() + offset, buffer, buffer + strlen(buffer));
			} else if (!strncmp(statement, "dcl_", 4)) {
				// Hateful strcmp logic is upside down, only output for ones we aren't already handling.
				if (strcmp(statement, "dcl_output") && strcmp(statement, "dcl_output_siv") && strcmp(statement, "dcl_globalFlags") &&
					strcmp(statement, "dcl_input_ps") && strcmp(statement, "dcl_input_ps_sgv") && strcmp(statement, "dcl_input_ps_siv")) {
					// Other declarations, unforeseen.
					sprintf_s(buffer, sizeof(buffer), "// Needs manual fix for instruction: \n");
					appendOutput(buffer);
					sprintf_s(buffer, sizeof(buffer), "// unknown dcl_: ");
					appendOutput(buffer);
					ASMLineOut(c, pos, size);
				}
			} else {
				switch (instr->eOpcode) {
				case OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT:
				case OPCODE_HS_CONTROL_POINT_PHASE:
				case OPCODE_HS_FORK_PHASE:
				case OPCODE_HS_JOIN_PHASE:
					// Just safely consume these opcodes if they made it into the instruction buffer
					break;
				case OPCODE_ITOF:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					sprintf_s(buffer, sizeof(buffer), "  %s = %s;\n", writeTarget(op1), ci(convertToInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_UTOF:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					sprintf_s(buffer, sizeof(buffer), "  %s = %s;\n", writeTarget(op1), ci(convertToUInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_MOV:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s;\n", writeTarget(op1), ci(op2).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(%s);\n", writeTarget(op1), ci(op2).c_str());
					appendOutput(buffer);
					if (op1[0] == 'o') {
						char *dotPos = strchr(op1, '.');
						if (dotPos)
							*dotPos = 0;
						if (!dotPos || dotPos[1] == 'x')
							mOutputRegisterValues[op1] = op2;
					}
					removeBoolean(op1);
					break;
				case OPCODE_RCP: emitUnaryMath("rcp", false); break;
				case OPCODE_NOT:
					remapTarget(op1);
					applySwizzle(op1, op2);
					sprintf_s(buffer, sizeof(buffer), "  %s = ~%s;\n", writeTarget(op1), ci(convertToInt(op2)).c_str());
					appendOutput(buffer);
					break;
				case OPCODE_INEG:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = -%s;\n", writeTarget(op1), ci(convertToInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_F32TOF16: emitUnaryMath("f32tof16"); break;
				case OPCODE_F16TOF32: emitUnaryMath("f16tof32"); break;
				case OPCODE_FRC: emitUnaryMath("frac"); break;
				case OPCODE_MUL:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]));
					mMulOperand = op3;
					mMulOperand2 = op2;
					mMulTarget = op1;
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s * %s;\n", writeTarget(op1), ci(op3).c_str(), ci(op2).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(%s * %s);\n", writeTarget(op1), ci(op3).c_str(),
								  ci(op2).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_IMUL:
					remapTarget(op2);
					applySwizzle(op2, op3, true);
					applySwizzle(op2, op4, true);
					mMulOperand = strncmp(op3, "int", 3) ? op3 : op4;
					sprintf_s(buffer, sizeof(buffer), "  %s = %s * %s;\n", writeTarget(op2), ci(convertToInt(op3)).c_str(),
							  ci(convertToInt(op4)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_DIV:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]));
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s / %s;\n", writeTarget(op1), ci(op2).c_str(), ci(op3).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(%s / %s);\n", writeTarget(op1), ci(op2).c_str(),
								  ci(op3).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

					// UDIV instruction also could damage the original register before finishing, giving wrong results.
					// e.g. udiv r0.x, r1.x, r0.x, r0.y
					// variant: udiv null, r1.xy, r3.zwzz, r2.zwzz
					// To fix this, we are using the temp variables declared at the top.
					// expected output:
					//   uiDest.x = (uint)r0.x / (uint)r0.y;
					//   r1.x = (uint)r0.x % (uint)r0.y;
					//   r0.x = uiDest.x;
					//
					// This will swizzle based on either op1 or op2, as long as it's not null. It's not clear whether
					// the swizzle is allowed to vary for each half of the instruction, like xy for /, zw for %.
					// To allow for that, we'll set the temp registers with full swizzle, then only use the specific
					// parts required for each half, as the safest approach. Might not generate udiv though.
					// Also removed saturate code, because udiv does not specify that.
					// Creates operand copies to applySwizzle to unchanged operands, as constant l values are otherwise damaged.
				case OPCODE_UDIV: {
					remapTarget(op1);
					remapTarget(op2);
					char divOut[opcodeSize] = "uiDest.xyzw";
					char *divSwiz = op1;
					char *remSwiz = op2;
					strcpy_s(op13, opcodeSize, op3);
					strcpy_s(op14, opcodeSize, op4);
					if (instr->asOperands[0].eType != OPERAND_TYPE_NULL) {
						applySwizzle(divSwiz, divOut, true);
						applySwizzle(divSwiz, fixImm(op13, instr->asOperands[2]), true);
						applySwizzle(divSwiz, fixImm(op14, instr->asOperands[3]), true);
						convertToUInt(op13);
						convertToUInt(op14);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s / %s;\n", divOut, ci(op13).c_str(), ci(op14).c_str());
						appendOutput(buffer);
					}
					if (instr->asOperands[1].eType != OPERAND_TYPE_NULL) {
						applySwizzle(remSwiz, fixImm(op3, instr->asOperands[2]), true);
						applySwizzle(remSwiz, fixImm(op4, instr->asOperands[3]), true);
						convertToUInt(op3);
						convertToUInt(op4);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s %% %s;\n", writeTarget(op2), ci(op3).c_str(), ci(op4).c_str());
						appendOutput(buffer);
					}
					if (instr->asOperands[0].eType != OPERAND_TYPE_NULL) {
						sprintf_s(buffer, sizeof(buffer), "  %s = %s;\n", writeTarget(op1), divOut);
						appendOutput(buffer);
					}
					removeBoolean(op1);
					removeBoolean(op2);
					break;
				}
				case OPCODE_ADD:
					// OPCODE_ADD is 0. Therefore, it is possible for us to arrive here if the line has simply not been parsed.
					// Let's make sure we are actually parsing an 'add' before we go any further.
					if (strncmp(statement, "add", 3)) {
						sprintf_s(buffer, sizeof(buffer), "// Unknown statement (maybe desync?): %s\n", statement);
						appendOutput(buffer);
						break;
					}
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]));
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s + %s;\n", writeTarget(op1), ci(op3).c_str(), ci(op2).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(%s + %s);\n", writeTarget(op1), ci(op3).c_str(),
								  ci(op2).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

					// IADD is apparently used by the fxc compiler in some cases like an AND operation.
					// After boolean operations, it will sum up two boolean operators. Since we are now
					// passing -1:0 instead of 1:0, it should work to just do the IADD operation, respecting
					// any negation the source applies. A common sequence, compiler trick is:
					//  lt r0.y, l(0.000000), r0.x
					//  lt r0.x, r0.x, l(0.000000)
					//  iadd r0.x, r0.x, -r0.y
					//  itof r0.x, r0.x
					//  mul r0.xyz, r0.xxxx, r1.xzwx
				case OPCODE_IADD:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					applySwizzle(op1, op3, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = %s + %s;\n", writeTarget(op1), ci(convertToInt(op2)).c_str(),
							  ci(convertToInt(op3)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

					// AND opcodes were generating bad code, as the hex constants were being converted badly.
					// The most common case was 0x3f800000 being converted directly to integer decimal of 1065353216, instead
					// of the most likely answer of floating point 1.0f. This also happened for conversion of Pi.
					// There are bitmasks used for AND, and those need to stay as Hex constants.
					// But anything used after IF statements/booleans, needs to be converted as float.
					// Rather than modify applySwizzle for this single opcode, it makes more sense to convert them here,
					// if they are to be used in boolean operations. We make a copy of the incoming operands, so that we
					// can applySwizzle in order to be able to look up isBoolean properly. Can be r3.xxxy, and becomes r3.xy.
					// That applySwizzle damages constants though, so if we are boolean, we'll use the original l() value.
				case OPCODE_AND:
					remapTarget(op1);
					strcpy_s(op12, op2);
					strcpy_s(op13, op3);
					applySwizzle(op1, op2);
					applySwizzle(op1, op3);
					if (isBoolean(op2) || isBoolean(op3)) {
						convertHexToFloat(op12);
						convertHexToFloat(op13);
						applySwizzle(op1, op12);
						applySwizzle(op1, op13);
						char *cmp = isBoolean(op2) ? op12 : op13;
						char *arg = isBoolean(op2) ? op13 : op12;
						sprintf_s(buffer, sizeof(buffer), "  %s = %s ? %s : 0;\n", writeTarget(op1), ci(cmp).c_str(), ci(arg).c_str());
						appendOutput(buffer);
					} else {
						sprintf_s(buffer, sizeof(buffer), "  %s = %s & %s;\n", writeTarget(op1), ci(convertToInt(op2)).c_str(),
								  ci(convertToInt(op3)).c_str());
						appendOutput(buffer);
					}
					break;

				case OPCODE_OR:
					remapTarget(op1);
					applySwizzle(op1, op2);
					applySwizzle(op1, op3);
					sprintf_s(buffer, sizeof(buffer), "  %s = %s | %s;\n", writeTarget(op1), ci(convertToInt(op2)).c_str(),
							  ci(convertToInt(op3)).c_str());
					appendOutput(buffer);
					break;

				case OPCODE_XOR:
					remapTarget(op1);
					applySwizzle(op1, op2);
					applySwizzle(op1, op3);
					sprintf_s(buffer, sizeof(buffer), "  %s = %s ^ %s;\n", writeTarget(op1), ci(convertToInt(op2)).c_str(),
							  ci(convertToInt(op3)).c_str());
					appendOutput(buffer);
					break;

					// Curiously enough, the documentation for ISHR and ISHL is wrong, and documents the parameters backwards.
					// http://msdn.microsoft.com/en-us/library/windows/desktop/hh447145(v=vs.85).aspx
					// This was proven by looking at actual game ASM, and trying to make the HLSL match the generated ASM.
					// It is the C standard of: shift-expression << additive-expression
					// So, we need op2 as the Shift-Expression, op3 as the # of bits to shift.
				case OPCODE_ISHR:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					applySwizzle(op1, op3, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = %s >> %s;\n", writeTarget(op1), ci(convertToUInt(op2)).c_str(),
							  ci(convertToInt(op3)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

				case OPCODE_ISHL:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					applySwizzle(op1, op3, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = %s << %s;\n", writeTarget(op1), ci(convertToUInt(op2)).c_str(),
							  ci(convertToInt(op3)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

					// USHR appears to be documented correctly. But this code was previously backwards.
				case OPCODE_USHR:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					applySwizzle(op1, op3, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = %s >> %s;\n", writeTarget(op1), ci(convertToUInt(op2)).c_str(),
							  ci(convertToUInt(op3)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

					// Add the Firstbit ops, because now Just Cause 3 uses them.
					// firstbit{_hi|_lo|_shi} dest[.mask], src0[.swizzle]
				case OPCODE_FIRSTBIT_HI:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = firstbithigh(%s);\n", writeTarget(op1), ci(convertToUInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_FIRSTBIT_LO:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = firstbitlow(%s);\n", writeTarget(op1), ci(convertToUInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_FIRSTBIT_SHI:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = firstbithigh(%s);\n", writeTarget(op1), ci(convertToInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

					// Code generation for this weird instruction is tuned to indent the way we want,
					// and still look like a single instruction. Looks like a less powerful version of rlwinm.
				case OPCODE_UBFE: {
					remapTarget(op1);
					removeBoolean(op1);
					applySwizzle(op1, op2);
					applySwizzle(op1, op3);
					applySwizzle(op1, op4);
					int idx = 0;
					char *pop1 = strrchr(op1, '.');
					*pop1 = 0;
					while (*++pop1) {
						sprintf_s(op5, sizeof(op5), "%s.%c", op1, *pop1);
						sprintf_s(buffer, sizeof(buffer),
								  "  if (%s == 0) %s = 0; else if (%s+%s < 32) { %s = (uint)%s << (32-(%s + %s)); %s = (uint)%s >> "
								  "(32-%s); } else %s = (uint)%s >> %s;\n",
								  ci(GetSuffix(op2, idx)).c_str(), writeTarget(op5), ci(GetSuffix(op2, idx)).c_str(),
								  ci(GetSuffix(op3, idx)).c_str(), writeTarget(op5), ci(GetSuffix(op4, idx)).c_str(),
								  ci(GetSuffix(op2, idx)).c_str(), ci(GetSuffix(op3, idx)).c_str(), writeTarget(op5), writeTarget(op5),
								  ci(GetSuffix(op2, idx)).c_str(), writeTarget(op5), ci(GetSuffix(op4, idx)).c_str(),
								  ci(GetSuffix(op3, idx)).c_str());
						appendOutput(buffer);
						++idx;
					}
					break;
				}
				case OPCODE_IBFE: {
					// Instruction for sign extending field extraction. Used in new versions of Dolphin
					// ibfe r0.xyzw, l(24, 24, 24, 24), l(0, 0, 0, 0), cb0[12].xyzw
					remapTarget(op1);
					removeBoolean(op1);
					applySwizzle(op1, op2, true);
					applySwizzle(op1, op3, true);
					applySwizzle(op1, op4);
					sprintf_s(buffer, sizeof(buffer),
							  "  %s = (%s == 0 ? 0 : (%s + %s < 32 ? (((int%s)%s << (32 - %s - %s)) >> (32 - %s)) : ((int%s)%s >> %s)));\n",
							  writeTarget(op1), ci(op2).c_str(), ci(op2).c_str(), ci(op3).c_str(), swizCount(op4).c_str(), ci(op4).c_str(),
							  ci(op2).c_str(), ci(op3).c_str(), ci(op2).c_str(), swizCount(op4).c_str(), ci(op4).c_str(), ci(op3).c_str());
					appendOutput(buffer);
					break;
				}
				case OPCODE_BFREV:
					remapTarget(op1);
					applySwizzle(op1, op2);
					sprintf_s(buffer, sizeof(buffer), "  %s = reversebits(%s);\n", writeTarget(op1), ci(convertToUInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_EXP: emitUnaryMath("exp2"); break;
				case OPCODE_LOG:
					emitUnaryMath("log2");
					break;

					// Opcodes for Sqrt, Min, Max, IMin, IMax code the generated HLSL directly
					// to avoid saturation name bugs (e.g. max_sat).
				case OPCODE_SQRT: emitUnaryMath("sqrt"); break;

				case OPCODE_MIN:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]));
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = min(%s, %s);\n", writeTarget(op1), ci(op3).c_str(), ci(op2).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(min(%s, %s));\n", writeTarget(op1), ci(op3).c_str(),
								  ci(op2).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

					// Missing opcode for UMin, used in Dragon Age
				case OPCODE_UMIN:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]));
					sprintf_s(buffer, sizeof(buffer), "  %s = min(%s, %s);\n", writeTarget(op1), ci(convertToUInt(op3)).c_str(),
							  ci(convertToUInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

					// Missing opcode for UMax, used in Witcher3
				case OPCODE_UMAX:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]));
					sprintf_s(buffer, sizeof(buffer), "  %s = max(%s, %s);\n", writeTarget(op1), ci(convertToUInt(op3)).c_str(),
							  ci(convertToUInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;

					// Atomic ops found in Compute Shaders (Witcher 3, Song of the Deep)
				case OPCODE_ATOMIC_AND: emitAtomicStub("InterlockedAnd(dest, value, orig_value);"); break;
				case OPCODE_ATOMIC_OR: emitAtomicStub("InterlockedOr(dest, value, orig_value);"); break;
				case OPCODE_ATOMIC_XOR: emitAtomicStub("InterlockedXor(dest, value, orig_value);"); break;
				case OPCODE_ATOMIC_CMP_STORE: emitAtomicStub("InterlockedCompareStore(dest, value, orig_value);"); break;
				case OPCODE_ATOMIC_IADD: emitAtomicStub("InterlockedAdd(dest, value, orig_value);"); break;
				case OPCODE_ATOMIC_IMAX: emitAtomicStub("InterlockedMax(dest, value, orig_value);"); break;
				case OPCODE_ATOMIC_IMIN: emitAtomicStub("InterlockedMin(dest, value, orig_value);"); break;
				case OPCODE_ATOMIC_UMAX: emitAtomicStub("InterlockedMax(dest, value, orig_value);"); break;
				case OPCODE_ATOMIC_UMIN: emitAtomicStub("InterlockedMin(dest, value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_ALLOC: emitAtomicStub("InterlockedExchange ?(dest, value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_CONSUME: emitAtomicStub("Interlocked... ?(dest, value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_IADD: emitAtomicStub("InterlockedAdd(dest, imm_value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_AND: emitAtomicStub("InterlockedAnd(dest, imm_value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_OR: emitAtomicStub("InterlockedOr(dest, imm_value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_XOR: emitAtomicStub("InterlockedXor(dest, imm_value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_EXCH: emitAtomicStub("InterlockedExchange(dest, imm_value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_CMP_EXCH:
					emitAtomicStub("InterlockedCompareExchange(dest, compare_value, imm_value, orig_value);");
					break;
				case OPCODE_IMM_ATOMIC_IMAX: emitAtomicStub("InterlockedMax(dest, imm_value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_IMIN: emitAtomicStub("InterlockedMin(dest, imm_value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_UMAX: emitAtomicStub("InterlockedMax(dest, imm_value, orig_value);"); break;
				case OPCODE_IMM_ATOMIC_UMIN: emitAtomicStub("InterlockedMin(dest, imm_value, orig_value);"); break;
				case OPCODE_MAX:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]));
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = max(%s, %s);\n", writeTarget(op1), ci(op3).c_str(), ci(op2).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(max(%s, %s));\n", writeTarget(op1), ci(op3).c_str(),
								  ci(op2).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_IMIN:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]), true);
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]), true);
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = min(%s, %s);\n", writeTarget(op1), ci(convertToInt(op3)).c_str(),
								  ci(convertToInt(op2)).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(min(%s, %s));\n", writeTarget(op1),
								  ci(convertToInt(op3)).c_str(), ci(convertToInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_IMAX:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]), true);
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]), true);
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = max(%s, %s);\n", writeTarget(op1), ci(convertToInt(op3)).c_str(),
								  ci(convertToInt(op2)).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(max(%s, %s));\n", writeTarget(op1),
								  ci(convertToInt(op3)).c_str(), ci(convertToInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_MAD:
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]));
					applySwizzle(op1, fixImm(op4, instr->asOperands[3]));
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s * %s + %s;\n", writeTarget(op1), ci(op2).c_str(), ci(op3).c_str(),
								  ci(op4).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(%s * %s + %s);\n", writeTarget(op1), ci(op2).c_str(),
								  ci(op3).c_str(), ci(op4).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_IMAD:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					applySwizzle(op1, op3, true);
					applySwizzle(op1, op4, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = mad(%s, %s, %s);\n", writeTarget(op1), ci(convertToInt(op2)).c_str(),
							  ci(convertToInt(op3)).c_str(), ci(convertToInt(op4)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_UMAD:
					remapTarget(op1);
					applySwizzle(op1, op2, true);
					applySwizzle(op1, op3, true);
					applySwizzle(op1, op4, true);
					sprintf_s(buffer, sizeof(buffer), "  %s = mad(%s, %s, %s);\n", writeTarget(op1), ci(convertToUInt(op2)).c_str(),
							  ci(convertToUInt(op3)).c_str(), ci(convertToUInt(op4)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_DP2: emitDot(instr, ".xy"); break;
				case OPCODE_DP3: emitDot(instr, ".xyz"); break;
				case OPCODE_DP4: emitDot(instr, ".xyzw"); break;
				case OPCODE_RSQ: emitUnaryMath("rsqrt"); break;

				// Double checked this while looking at other round bugs. Looks correct to use 'floor'.
				case OPCODE_ROUND_NI: emitUnaryMath("floor"); break;

				case OPCODE_ROUND_PI: emitUnaryMath("ceil"); break;

				// Needs to round toward zero (Trunc), Round can go larger. Trunc(1.6)->1.0, Round(1.6)->2.0
				case OPCODE_ROUND_Z: emitUnaryMath("trunc"); break;

				// Round Nearest Even (HLSL round)
				case OPCODE_ROUND_NE: emitUnaryMath("round"); break;

				case OPCODE_FTOI:
					remapTarget(op1);
					applySwizzle(op1, op2);
					sprintf_s(buffer, sizeof(buffer), "  %s = %s;\n", writeTarget(op1), ci(castToInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_FTOU:
					remapTarget(op1);
					applySwizzle(op1, op2);
					sprintf_s(buffer, sizeof(buffer), "  %s = %s;\n", writeTarget(op1), ci(castToUInt(op2)).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				case OPCODE_SINCOS:
					remapTarget(op1);
					remapTarget(op2);
					if (!strncmp(op1, "null", 4))
						applySwizzle(op2, op3);
					else
						applySwizzle(op1, op3);
					if (!strncmp(op1, "null", 4))
						sprintf_s(buffer, sizeof(buffer), "  %s = cos(%s);\n", writeTarget(op2), ci(op3).c_str());
					else if (!strncmp(op2, "null", 4))
						sprintf_s(buffer, sizeof(buffer), "  %s = sin(%s);\n", writeTarget(op1), ci(op3).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  sincos(%s, %s, %s);\n", ci(op3).c_str(), writeTarget(op1), writeTarget(op2));
					appendOutput(buffer);
					removeBoolean(op1);
					removeBoolean(op2);
					break;

				// Failing case of: "movc_sat r2.xyzw, r2.xxxx, r7.xyzw, r4.xyzw"
				// Unrolling damaged r2.x at the first line and then reused it in each.
				// Changed to single vector assignment: r2.xyzw = saturate(r2.xxxx ? r7.xyzw : r4.xyzw);
				case OPCODE_MOVC: {
					remapTarget(op1);
					applySwizzle(op1, fixImm(op2, instr->asOperands[1]));
					applySwizzle(op1, fixImm(op3, instr->asOperands[2]));
					applySwizzle(op1, fixImm(op4, instr->asOperands[3]));
					if (!instr->bSaturate)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s ? %s : %s;\n", writeTarget(op1), ci(op2).c_str(), ci(op3).c_str(),
								  ci(op4).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = saturate(%s ? %s : %s);\n", writeTarget(op1), ci(op2).c_str(),
								  ci(op3).c_str(), ci(op4).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}

				// Big change to all these boolean test opcodes. All were unrolled and generated a code line per component.
				// To make the boolean tests work correctly for AND, these were rolled back into one, and added to the boolean
				// set list as a complete operand, like 'r0.xyw'.
				case OPCODE_NE: emitComparison(instr, "!="); break;
				case OPCODE_INE: emitComparison(instr, "!=", true, false); break;
				case OPCODE_EQ: emitComparison(instr, "=="); break;
				case OPCODE_IEQ: emitComparison(instr, "==", true, false); break;
				case OPCODE_LT: emitComparison(instr, "<"); break;
				case OPCODE_ILT: emitComparison(instr, "<", true, false); break;
				case OPCODE_ULT: emitComparison(instr, "<", false, true); break;
				case OPCODE_GE: emitComparison(instr, ">="); break;
				case OPCODE_IGE: emitComparison(instr, ">=", true, false); break;
				case OPCODE_UGE: emitComparison(instr, ">=", false, true); break;

				// Switch statement in HLSL was missing. Added because AC4 uses it.
				case OPCODE_SWITCH:
					sprintf_s(buffer, sizeof(buffer), "  switch (%s) {\n", ci(op1).c_str());
					appendOutput(buffer);
					break;
				case OPCODE_CASE:
					sprintf_s(buffer, sizeof(buffer), "  case %s :", ci(op1).substr(2, 1).c_str());
					appendOutput(buffer);
					break;
				case OPCODE_ENDSWITCH:
					sprintf_s(buffer, sizeof(buffer), "  }\n");
					appendOutput(buffer);
					break;
				case OPCODE_DEFAULT:
					sprintf_s(buffer, sizeof(buffer), "  default :\n");
					appendOutput(buffer);
					break;

				case OPCODE_IF:
					applySwizzle(".x", op1);
					if (instr->eBooleanTestType == INSTRUCTION_TEST_ZERO)
						sprintf_s(buffer, sizeof(buffer), "  if (%s == 0) {\n", ci(op1).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  if (%s != 0) {\n", ci(op1).c_str());
					appendOutput(buffer);
					break;
				case OPCODE_ELSE:
					sprintf_s(buffer, sizeof(buffer), "  } else {\n");
					appendOutput(buffer);
					break;
				case OPCODE_ENDIF:
					sprintf_s(buffer, sizeof(buffer), "  }\n");
					appendOutput(buffer);
					break;

				case OPCODE_LOOP:
					sprintf_s(buffer, sizeof(buffer), "  while (true) {\n");
					appendOutput(buffer);
					break;
				case OPCODE_BREAK:
					sprintf_s(buffer, sizeof(buffer), "  break;\n");
					appendOutput(buffer);
					break;
				case OPCODE_BREAKC:
					applySwizzle(".x", op1);
					if (instr->eBooleanTestType == INSTRUCTION_TEST_ZERO)
						sprintf_s(buffer, sizeof(buffer), "  if (%s == 0) break;\n", ci(op1).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  if (%s != 0) break;\n", ci(op1).c_str());
					appendOutput(buffer);
					break;
				case OPCODE_CONTINUE:
					sprintf_s(buffer, sizeof(buffer), "  continue;\n");
					appendOutput(buffer);
					break;
				case OPCODE_CONTINUEC:
					applySwizzle(".x", op1);
					if (instr->eBooleanTestType == INSTRUCTION_TEST_ZERO)
						sprintf_s(buffer, sizeof(buffer), "  if (%s == 0) continue;\n", ci(op1).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  if (%s != 0) continue;\n", ci(op1).c_str());
					appendOutput(buffer);
					break;
				case OPCODE_ENDLOOP:
					sprintf_s(buffer, sizeof(buffer), "  }\n");
					appendOutput(buffer);
					break;

				// Found in Witcher3 Compute Shaders
				case OPCODE_SYNC:
					if (!strcmp(statement, "sync_g_t"))
						sprintf_s(buffer, sizeof(buffer), "  GroupMemoryBarrierWithGroupSync();\n");
					else
						sprintf_s(buffer, sizeof(buffer), "  Unknown sync instruction;\n");
					appendOutput(buffer);
					break;

				case OPCODE_SWAPC: {
					remapTarget(op1);
					remapTarget(op2);
					removeBoolean(op1); // The code damages the op1, op2 below.
					removeBoolean(op2);
					applySwizzle(op1, op3);
					applySwizzle(op1, op4);
					applySwizzle(op1, op5);
					int idx = 0;
					char *pop1 = strrchr(op1, '.');
					*pop1 = 0;
					char *pop2 = strrchr(op2, '.');
					if (pop2)
						*pop2 = 0;
					char *pop3 = strrchr(op3, '.');
					if (pop3)
						*pop3 = 0;
					while (*++pop1) {
						sprintf_s(op6, sizeof(op6), "%s.%c", op1, *pop1);
						if (pop2)
							sprintf_s(op7, sizeof(op7), "%s.%c", op2, *++pop2);
						else
							sprintf_s(op7, sizeof(op7), "%s", op2);
						if (pop3)
							sprintf_s(op8, sizeof(op8), "%s.%c", op3, *++pop3);
						else
							sprintf_s(op8, sizeof(op8), "%s", op3);
						// FIXME: May need fixup for read from constant buffer of unidentified type
						sprintf_s(buffer, sizeof(buffer), "  %s = (int)%s ? %s : %s; %s = (int)%s ? %s : %s;\n", writeTarget(op6),
								  ci(op8).c_str(), ci(GetSuffix(op5, idx)).c_str(), ci(GetSuffix(op4, idx)).c_str(), writeTarget(op7),
								  ci(op8).c_str(), ci(GetSuffix(op4, idx)).c_str(), ci(GetSuffix(op5, idx)).c_str());
						appendOutput(buffer);
						++idx;
					}
					break;
				}

				// This generated code needed to change because the fxc compiler generates bad code when
				// using the sample that they specify in the documentation at:
				// http://msdn.microsoft.com/en-us/library/windows/desktop/hh446837(v=vs.85).aspx
				// I worked out the alternate technique that works for all, and does not tickle
				// the bug that treats 0x80000000 as "-0" as a uint, where it should not exist.
				case OPCODE_BFI: {
					remapTarget(op1);
					removeBoolean(op1);
					applySwizzle(op1, op2);
					applySwizzle(op1, op3);
					applySwizzle(op1, op4);
					applySwizzle(op1, op5);
					int idx = 0;
					char *pop1 = strrchr(op1, '.');
					*pop1 = 0;
					while (*++pop1) {
						sprintf_s(op6, sizeof(op6), "%s.%c", op1, *pop1);
						sprintf_s(buffer, sizeof(buffer),
								  "  bitmask.%c = ((~(-1 << %s)) << %s) & 0xffffffff;  %s = (((uint)%s << %s) & bitmask.%c) | ((uint)%s & "
								  "~bitmask.%c);\n",
								  *pop1, ci(GetSuffix(op2, idx)).c_str(), ci(GetSuffix(op3, idx)).c_str(), writeTarget(op6),
								  ci(GetSuffix(op4, idx)).c_str(), ci(GetSuffix(op3, idx)).c_str(), *pop1, ci(GetSuffix(op5, idx)).c_str(),
								  *pop1);
						appendOutput(buffer);
						++idx;
					}
					break;
				}

				// Was missing the sample_aoffimmi variant. Added as matching sample_b type. Used in FC4.
				case OPCODE_SAMPLE: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					int textureId = parseResourceIndex(op3);
					int samplerId = parseResourceIndex(op4);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.Sample(%s, %s)%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), strrchr(op3, '.'));
					else {
						int offsetx = 0, offsety = 0, offsetz = 0;
						sscanf_s(statement, "sample_aoffimmi(%d,%d,%d", &offsetx, &offsety, &offsetz);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.Sample(%s, %s, int2(%d, %d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), offsetx, offsety,
								  strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}

				// Missing opcode for WatchDogs. Very similar to SAMPLE_L, so copied from there.
				case OPCODE_SAMPLE_B: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					applySwizzle(".x", fixImm(op5, instr->asOperands[4]));
					int textureId = parseResourceIndex(op3);
					int samplerId = parseResourceIndex(op4);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleBias(%s, %s, %s)%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), ci(op5).c_str(),
								  strrchr(op3, '.'));
					else {
						int offsetx = 0, offsety = 0, offsetz = 0;
						sscanf_s(statement, "sample_b_aoffimmi_indexable(%d,%d,%d", &offsetx, &offsety, &offsetz);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleBias(%s, %s, %s, int2(%d, %d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), ci(op5).c_str(),
								  offsetx, offsety, strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				case OPCODE_SAMPLE_L: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					applySwizzle(".x", fixImm(op5, instr->asOperands[4]));
					int textureId = parseResourceIndex(op3);
					int samplerId = parseResourceIndex(op4);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleLevel(%s, %s, %s)%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), ci(op5).c_str(),
								  strrchr(op3, '.'));
					else {
						int offsetx = 0, offsety = 0, offsetz = 0;
						sscanf_s(statement, "sample_l_aoffimmi_indexable(%d,%d,%d", &offsetx, &offsety, &offsetz);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleLevel(%s, %s, %s, int2(%d, %d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), ci(op5).c_str(),
								  offsetx, offsety, strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				case OPCODE_SAMPLE_D: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					applySwizzle(op1, fixImm(op5, instr->asOperands[4]));
					applySwizzle(op1, fixImm(op6, instr->asOperands[5]));
					int textureId = parseResourceIndex(op3);
					int samplerId = parseResourceIndex(op4);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleGrad(%s, %s, %s, %s)%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), ci(op5).c_str(),
								  ci(op6).c_str(), strrchr(op3, '.'));
					else {
						int offsetx = 0, offsety = 0, offsetz = 0;
						sscanf_s(statement, "sample_d_aoffimmi_indexable(%d,%d,%d", &offsetx, &offsety, &offsetz);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleGrad(%s, %s, %s, %s, int2(%d, %d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), ci(op5).c_str(),
								  ci(op6).c_str(), offsetx, offsety, strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				case OPCODE_SAMPLE_C: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					applySwizzle(".x", fixImm(op5, instr->asOperands[4]));
					int textureId = parseResourceIndex(op3);
					int samplerId = parseResourceIndex(op4);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleCmp(%s, %s, %s)%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerComparisonNames[samplerId].c_str(), ci(op2).c_str(),
								  ci(op5).c_str(), strrchr(op3, '.'));
					else {
						int offsetx = 0, offsety = 0, offsetz = 0;
						sscanf_s(statement, "sample_c_aoffimmi_indexable(%d,%d,%d", &offsetx, &offsety, &offsetz);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleCmp(%s, %s, %s, int2(%d, %d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerComparisonNames[samplerId].c_str(), ci(op2).c_str(),
								  ci(op5).c_str(), offsetx, offsety, strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				case OPCODE_SAMPLE_C_LZ: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					applySwizzle(".x", fixImm(op5, instr->asOperands[4]));
					int textureId = parseResourceIndex(op3);
					int samplerId = parseResourceIndex(op4);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleCmpLevelZero(%s, %s, %s)%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerComparisonNames[samplerId].c_str(), ci(op2).c_str(),
								  ci(op5).c_str(), strrchr(op3, '.'));
					else {
						int offsetx = 0, offsety = 0, offsetz = 0;
						sscanf_s(statement, "sample_c_lz_aoffimmi_indexable(%d,%d,%d", &offsetx, &offsety, &offsetz);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.SampleCmpLevelZero(%s, %s, %s, int2(%d, %d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerComparisonNames[samplerId].c_str(), ci(op2).c_str(),
								  ci(op5).c_str(), offsetx, offsety, strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				// This opcode was missing, and used in WatchDogs.
				// expected code "samplepos r0.xy, t1.xyxx, v1.x" -> "r0.xy = t1.GetSamplePosition(v1.x);"
				case OPCODE_SAMPLE_POS: {
					remapTarget(op1);
					applySwizzle(op1, op2);
					applySwizzle(op1, op3);
					int textureId = parseResourceIndex(op2);
					sprintf_s(buffer, sizeof(buffer), "  %s = %s.GetSamplePosition(%s);\n", writeTarget(op1),
							  mTextureNames[textureId].c_str(), ci(op3).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				// Missing opcode, used in FC4. Similar to 'sample'
				// lod dest[.mask], srcAddress[.swizzle], srcResource[.swizzle], srcSampler
				// ret Object.CalculateLevelOfDetail(sampler_state S, float x);
				// "lod r0.x, r0.xyzx, t2.y, s2" -> "r0.x = t2.CalculateLevelOfDetailUnclamped(s2, r0.xyz);"
				// CalculateLevelOfDetailUnclamped compiles to t2.y,
				// CalculateLevelOfDetail compiles to t2.x
				case OPCODE_LOD: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					int textureId = parseResourceIndex(op3);
					int samplerId = parseResourceIndex(op4);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					char *clamped = strrchr(op3, '.') + 1;
					if (*clamped == 'x')
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.CalculateLevelOfDetail(%s, %s);\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.CalculateLevelOfDetailUnclamped(%s, %s);\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str());
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				case OPCODE_GATHER4: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					int textureId = parseResourceIndex(op3);
					int samplerId = parseResourceIndex(op4);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.Gather(%s, %s)%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), strrchr(op3, '.'));
					else {
						int offsetx = 0, offsety = 0, offsetz = 0;
						sscanf_s(statement, "gather4_aoffimmi_indexable(%d,%d,%d", &offsetx, &offsety, &offsetz);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.Gather(%s, %s, int2(%d, %d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), offsetx, offsety,
								  strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				case OPCODE_GATHER4_C: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					int textureId = parseResourceIndex(op3);
					int samplerId = parseResourceIndex(op4);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.GatherCmp(%s, %s, %s)%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerComparisonNames[samplerId].c_str(), ci(op2).c_str(),
								  ci(op5).c_str(), strrchr(op3, '.'));
					else {
						int offsetx = 0, offsety = 0, offsetz = 0;
						sscanf_s(statement, "gather4_c_aoffimmi_indexable(%d,%d,%d", &offsetx, &offsety, &offsetz);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.GatherCmp(%s, %s, %s, int2(%d,%d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), mSamplerComparisonNames[samplerId].c_str(), ci(op2).c_str(),
								  ci(op5).c_str(), offsetx, offsety, strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				// Add the Gather4_PO opcodes for Dragon Age. Copied from Gather4.
				//   gather4_po dest[.mask], srcAddress[.swizzle], srcOffset[.swizzle], srcResource[.swizzle], srcSampler[.select_component]
				//   output.color = texture2d.Gather(samplerState, input.texcoord, int2(0,0));
				case OPCODE_GATHER4_PO: {
					remapTarget(op1);
					applySwizzle(op1, op2);
					applySwizzle(op1, op3);
					applySwizzle(op1, op4);
					int textureId = parseResourceIndex(op4);
					int samplerId = parseResourceIndex(op5);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					sprintf_s(buffer, sizeof(buffer), "  %s = %s.Gather(%s, %s, %s)%s;\n", writeTarget(op1),
							  mTextureNames[textureId].c_str(), mSamplerNames[samplerId].c_str(), ci(op2).c_str(), ci(op3).c_str(),
							  strrchr(op4, '.'));
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				// gather4_po_c dest[.mask], srcAddress[.swizzle], srcOffset[.swizzle], srcResource[.swizzle], srcSampler[.R],
				// srcReferenceValue
				case OPCODE_GATHER4_PO_C: {
					remapTarget(op1);
					applySwizzle(op1, op2);
					applySwizzle(op1, op3);
					applySwizzle(op1, op4);
					int textureId = parseResourceIndex(op4);
					int samplerId = parseResourceIndex(op5);
					truncateTexturePos(op2, mTextureType[textureId].c_str());
					sprintf_s(buffer, sizeof(buffer), "  %s = %s.GatherCmp(%s, %s, %s, %s)%s;\n", writeTarget(op1),
							  mTextureNames[textureId].c_str(), mSamplerComparisonNames[samplerId].c_str(), ci(op2).c_str(),
							  ci(op6).c_str(), ci(op3).c_str(), strrchr(op4, '.'));
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				// For the _aoffimmi format, this was picking up 0..15, instead of the necessary -8..7
				// It's a 4 bit number being used in an int, hence missing negatives.
				// Fixed both _LD and LD_MS
				case OPCODE_LD: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					int textureId = parseResourceIndex(op3);
					truncateTextureLoadPos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.Load(%s)%s;\n", writeTarget(op1), mTextureNames[textureId].c_str(),
								  ci(op2).c_str(), strrchr(op3, '.'));
					else {
						int offsetU = 0, offsetV = 0, offsetW = 0;
						sscanf_s(statement, "ld_aoffimmi(%d,%d,%d", &offsetU, &offsetV, &offsetW);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.Load(%s, int3(%d, %d, %d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), ci(op2).c_str(), offsetU, offsetV, offsetW, strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}
				case OPCODE_LD_MS: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					applySwizzle(".x", fixImm(op4, instr->asOperands[3]), true);
					int textureId = parseResourceIndex(op3);
					truncateTextureLoadPos(op2, mTextureType[textureId].c_str());
					if (!instr->bAddressOffset)
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.Load(%s, %s)%s;\n", writeTarget(op1), mTextureNames[textureId].c_str(),
								  ci(op2).c_str(), ci(op4).c_str(), strrchr(op3, '.'));
					else {
						int offsetU = 0, offsetV = 0, offsetW = 0;
						sscanf_s(statement, "ld_aoffimmi(%d,%d,%d", &offsetU, &offsetV, &offsetW);
						sprintf_s(buffer, sizeof(buffer), "  %s = %s.Load(%s, %s, int3(%d, %d, %d))%s;\n", writeTarget(op1),
								  mTextureNames[textureId].c_str(), ci(op2).c_str(), ci(op4).c_str(), offsetU, offsetV, offsetW,
								  strrchr(op3, '.'));
					}
					appendOutput(buffer);
					removeBoolean(op1);
					break;
				}

				// Example from Mordor, with bizarre struct offsets:
				// struct BufferSrc
				// {
				//	float3 vposition; // offset: 0
				//	float3 vvelocity; // offset: 12
				//	float ftime; // offset: 24
				//	float fuserdata; // offset: 28
				// }; // offset: 0 size: 32
				//
				// StructuredBuffer<BufferSrc> BufferSrc_SB : register(t0);
				//
				// Working fxc code (unrolled is necessary):
				// ld_structured_indexable(structured_buffer, stride=32)(mixed,mixed,mixed,mixed) r3.xyzw, v0.x, l(16), t0.xyzw
				// r3.x = BufferSrc_SB[v0.x].vvelocity.y;
				// r3.y = BufferSrc_SB[v0.x].vvelocity.z;
				// r3.z = BufferSrc_SB[v0.x].ftime.x;
				// r3.w = BufferSrc_SB[v0.x].fuserdata.x;
				case OPCODE_LD_STRUCTURED: {
					string dst0, srcAddress, srcByteOffset, src0;
					string swiz;
					ResourceBinding *bindings = shader->sInfo->psResourceBindings;
					if (bindings == NULL) {
						sprintf_s(buffer, sizeof(buffer), "// Missing reflection info for shader. No names possible.\n");
						appendOutput(buffer);
						src0 = "no_StructuredBufferName";
						srcAddress = "no_srcAddressRegister";
						srcByteOffset = "no_srcByteOffsetName";
					} else {
						src0 = bindings->Name;
						srcAddress = instr->asOperands[1].specialName;
						srcByteOffset = instr->asOperands[2].specialName;
					}
					dst0 = "r" + std::to_string(instr->asOperands[0].ui32RegisterNumber);
					sprintf_s(buffer, sizeof(buffer), "// Known bad code for instruction (needs manual fix):\n");
					appendOutput(buffer);
					ASMLineOut(c, pos, size);

					for (int component = 0; component < 4; component++) {
						if (instr->asOperands[0].ui32CompMask & (1 << component)) {
							switch (component) {
							case 3: swiz = "w"; break;
							case 2: swiz = "z"; break;
							case 1: swiz = "y"; break;
							case 0:
							default: swiz = "x"; break;
							}
							sprintf_s(buffer, sizeof(buffer), "%s.%s = %s[%s].%s.swiz;\n", dst0.c_str(), swiz.c_str(), src0.c_str(),
									  srcAddress.c_str(), srcByteOffset.c_str());
							appendOutput(buffer);
						}
					}
					removeBoolean(op1);
					break;
				}

				case OPCODE_STORE_STRUCTURED: {
					remapTarget(op1);
					applySwizzle(".xyzw", op2);
					applySwizzle(op1, op3);
					int textureId = parseResourceIndex(op4);
					sprintf_s(buffer, sizeof(buffer), "  %s[%s].%s = %s;\n", mTextureNames[textureId].c_str(), ci(op2).c_str(),
							  ci(op3).c_str(), writeTarget(op1));
					appendOutput(buffer);
					break;
				}

				// Missing opcodes for SM5. Not implemented yet, but we want to generate some sort of code, in case
				// these are used in needed shaders. That way we can hand edit the shader to make it usable, until this is completed.
				case OPCODE_STORE_UAV_TYPED:
				case OPCODE_LD_UAV_TYPED:
				case OPCODE_LD_RAW:
				case OPCODE_STORE_RAW: {
					sprintf_s(buffer, sizeof(buffer), "// No code for instruction (needs manual fix):\n");
					appendOutput(buffer);
					ASMLineOut(c, pos, size);
					break;
				}

				case OPCODE_DISCARD:
					applySwizzle(".x", op1);
					if (instr->eBooleanTestType == INSTRUCTION_TEST_ZERO)
						sprintf_s(buffer, sizeof(buffer), "  if (%s == 0) discard;\n", ci(op1).c_str());
					else
						sprintf_s(buffer, sizeof(buffer), "  if (%s != 0) discard;\n", ci(op1).c_str());
					appendOutput(buffer);
					break;

				// The GetDimensions can also see a 4 parameter version in the immediate case.
				// https://msdn.microsoft.com/en-us/library/windows/desktop/hh447214(v=vs.85).aspx
				//
				// resinfo[_uint|_rcpFloat] dest[.mask], srcMipLevel.select_component, srcResource[.swizzle]
				// In different variants based on input texture, becomes:
				// void Object.GetDimensions(UINT MipLevel, typeX Width, typeX Height, typeX Elements, typeX Depth, typeX NumberOfLevels,
				// typeX NumberOfSamples);
				//
				// We only see the immediate version l(0) in use, like:
				//  resinfo_indexable(texture2d)(uint, uint, uint, uint)_uint r2.zw, l(0), t5.zwxy
				//   Becomes 2 param version:   SectorAtlasTexture_UINT_TextureObject.GetDimensions(r2.z, r2.w);
				//  resinfo_indexable(texture2dms)(float, float, float, float)_uint r0.xy, l(0), t0.xyzw
				//   Becomes 3 param version:   DepthVPSampler_TextureObject.GetDimensions(r0.x, r0.y, bitmask.x);
				//  resinfo_indexable(texture2darray)(float, float, float, float) r0.xy, l(0), t0.xyzw
				//
				// So, we'll only handle that immediate for now, and generate syntax errors if we see any other variant.
				// We don't want to knowingly generate code that compiles, but has errors. Includes _rcpFloat as unknown.
				case OPCODE_RESINFO: {
					remapTarget(op1);
					bool unknownVariant = true;
					Operand output = instr->asOperands[0];
					Operand constZero = instr->asOperands[1];
					Operand texture = instr->asOperands[2];
					RESINFO_RETURN_TYPE returnType = instr->eResInfoReturnType;
					int texReg = texture.ui32RegisterNumber;
					ResourceBinding bindInfo;
					ResourceBinding *bindInfoPtr = &bindInfo;

					memset(&bindInfo, 0, sizeof(bindInfo));
					int bindstate = GetResourceFromBindingPoint(RTYPE_TEXTURE, texReg, shader->sInfo, &bindInfoPtr);
					bool bindStripped = (bindstate == 0);

					if (bindStripped) {
						// In the case where the reflection information has been stripped from the headers,
						// we are left with only the text line itself. Try to parse the text for variants
						// we know, and add them to the bindInfo.
						// e.g. from Batman and Witcher3:
						//  resinfo_indexable(texture2d)(float,float,float,float)_uint r1.yw, l(0), t3.zxwy
						char texType[opcodeSize];
						char retType[opcodeSize];
						int numInfo = sscanf_s(statement, "resinfo_indexable(%[^)])%s", texType, opcodeSize, retType, opcodeSize);
						bool isConstant = (!strcmp(op2, "l(0),"));

						if ((numInfo == 2) && isConstant) {
							constZero.eType = OPERAND_TYPE_IMMEDIATE32;
							constZero.afImmediates[0] = 0.0;
							if (!strcmp(texType, "texture2d"))
								bindInfoPtr->eDimension = REFLECT_RESOURCE_DIMENSION_TEXTURE2D;
							else if (!strcmp(texType, "texture2dms"))
								bindInfoPtr->eDimension = REFLECT_RESOURCE_DIMENSION_TEXTURE2DMS;
							else if (!strcmp(texType, "texture2darray"))
								bindInfoPtr->eDimension = REFLECT_RESOURCE_DIMENSION_TEXTURE2DARRAY;

							bindInfoPtr->Name = string(op3, strrchr(op3, '.'));
							if (strstr(retType, "_uint"))
								returnType = RESINFO_INSTRUCTION_RETURN_UINT;
							else
								returnType = RESINFO_INSTRUCTION_RETURN_FLOAT;
							texture.eType = OPERAND_TYPE_RESOURCE;
							bindStripped = false;
						}
					}

					// We only presently handle the float and _uint return types, and the const 0 mode.
					// And the texture2d and textures2dms types. That's all we've seen so far.
					if ((constZero.eType == OPERAND_TYPE_IMMEDIATE32) && (constZero.afImmediates[0] == 0.0) &&
						(returnType == RESINFO_INSTRUCTION_RETURN_UINT || returnType == RESINFO_INSTRUCTION_RETURN_FLOAT) &&
						texture.eType == OPERAND_TYPE_RESOURCE && !bindStripped) {

						if (bindInfoPtr->eDimension == REFLECT_RESOURCE_DIMENSION_TEXTURE2D) {
							if (returnType == RESINFO_INSTRUCTION_RETURN_UINT)
								sprintf_s(buffer, sizeof(buffer), "  %s.GetDimensions(0, uiDest.x, uiDest.y, uiDest.z);\n",
										  bindInfoPtr->Name.c_str());
							else
								sprintf_s(buffer, sizeof(buffer), "  %s.GetDimensions(0, fDest.x, fDest.y, fDest.z);\n",
										  bindInfoPtr->Name.c_str());
							appendOutput(buffer);
							unknownVariant = false;
						} else if (bindInfoPtr->eDimension == REFLECT_RESOURCE_DIMENSION_TEXTURE2DMS) {
							if (returnType == RESINFO_INSTRUCTION_RETURN_UINT)
								sprintf_s(buffer, sizeof(buffer), "  %s.GetDimensions(uiDest.x, uiDest.y, uiDest.z);\n",
										  bindInfoPtr->Name.c_str());
							else
								sprintf_s(buffer, sizeof(buffer), "  %s.GetDimensions(fDest.x, fDest.y, fDest.z);\n",
										  bindInfoPtr->Name.c_str());
							appendOutput(buffer);
							unknownVariant = false;
						} else if (bindInfoPtr->eDimension == REFLECT_RESOURCE_DIMENSION_TEXTURE2DARRAY) {
							if (returnType == RESINFO_INSTRUCTION_RETURN_UINT)
								sprintf_s(buffer, sizeof(buffer), "  %s.GetDimensions(0, uiDest.x, uiDest.y, uiDest.z, uiDest.w);\n",
										  bindInfoPtr->Name.c_str());
							else
								sprintf_s(buffer, sizeof(buffer), "  %s.GetDimensions(0, fDest.x, fDest.y, fDest.z, fDest.w);\n",
										  bindInfoPtr->Name.c_str());
							appendOutput(buffer);
							unknownVariant = false;
						}

						// For the output, we saw a r3.xyzw which makes no sense for this instruction.
						// Not sure this is fully correct, but the goal here is to apply the swizzle from the op3, which is the texture
						// register, as the pieces that are valid to copy to the op1 output.
						// The op3 texture swizzle can determine which components to use, and what order.
						// The op1 output determines which ones are valid for output.
						if (returnType == RESINFO_INSTRUCTION_RETURN_UINT) {
							char dest[opcodeSize] = "uiDest.xyzw";
							applySwizzle(op3, dest);
							applySwizzle(op1, dest);
							sprintf_s(buffer, sizeof(buffer), "  %s = %s;\n", op1, dest);
							appendOutput(buffer);
						} else {
							char dest[opcodeSize] = "fDest.xyzw";
							applySwizzle(op3, dest);
							applySwizzle(op1, dest);
							sprintf_s(buffer, sizeof(buffer), "  %s = %s;\n", op1, dest);
							appendOutput(buffer);
						}
					}

					if (unknownVariant) {
						// Completely new variant, write out reminder for manual fix
						string line = string(c + pos);
						line = line.substr(0, line.find('\n'));
						sprintf_s(buffer, sizeof(buffer),
								  "// Unknown use of GetDimensions for resinfo_ from missing reflection info, need manual fix.\n");
						appendOutput(buffer);
						sprintf_s(buffer, sizeof(buffer), "// %s\n", line.c_str());
						appendOutput(buffer);
						sprintf_s(buffer, sizeof(buffer), "// Example for texture2d type, uint return:\n");
						appendOutput(buffer);
						sprintf_s(buffer, sizeof(buffer), "tx.GetDimensions(0, uiDest.x, uiDest.y, uiDest.z);\n");
						appendOutput(buffer);
						sprintf_s(buffer, sizeof(buffer), "rx = uiDest;\n");
						appendOutput(buffer);
					}
					break;
				}

				case OPCODE_EVAL_SAMPLE_INDEX:
					remapTarget(op1);
					applySwizzle(op1, op2);
					applySwizzle(".x", fixImm(op3, instr->asOperands[2]), true);
					sprintf_s(buffer, sizeof(buffer), "  %s = EvaluateAttributeAtSample(%s, %s);\n", writeTarget(op1), op2, op3);
					appendOutput(buffer);
					break;

				case OPCODE_EVAL_CENTROID:
					remapTarget(op1);
					applySwizzle(op1, op2);
					sprintf_s(buffer, sizeof(buffer), "  %s = EvaluateAttributeCentroid(%s);\n", writeTarget(op1), op2);
					appendOutput(buffer);
					break;

				case OPCODE_EVAL_SNAPPED:
					remapTarget(op1);
					applySwizzle(op1, op2);
					applySwizzle(".xy", fixImm(op3, instr->asOperands[2]), true);
					sprintf_s(buffer, sizeof(buffer), "  %s = EvaluateAttributeSnapped(%s, %s);\n", writeTarget(op1), op2, op3);
					appendOutput(buffer);
					break;

				case OPCODE_DERIV_RTX_COARSE: emitDeriv("ddx_coarse"); break;
				case OPCODE_DERIV_RTX_FINE: emitDeriv("ddx_fine"); break;
				case OPCODE_DERIV_RTY_COARSE: emitDeriv("ddy_coarse"); break;
				case OPCODE_DERIV_RTY_FINE: emitDeriv("ddy_fine"); break;
				case OPCODE_DERIV_RTX: emitDeriv("ddx"); break;
				case OPCODE_DERIV_RTY: emitDeriv("ddy"); break;

				case OPCODE_RET:
					if (mShaderType.substr(0, 3) == "hs_") {
						// Do not emit premature "return;" inside Hull Shader phase blocks.
					} else {
						sprintf_s(buffer, sizeof(buffer), "  return;\n");
						appendOutput(buffer);
					}
					break;

				// Missing opcode needed for WatchDogs. Used as "retc_nz r0.x"
				case OPCODE_RETC:
					applySwizzle(".x", op1);
					if (mShaderType.substr(0, 3) == "hs_") {
						const char *retVal = (mHSPhase == 1) ? "output" : "patchConsts";
						if (instr->eBooleanTestType == INSTRUCTION_TEST_ZERO)
							sprintf_s(buffer, sizeof(buffer), "  if (%s == 0) return %s;\n", ci(op1).c_str(), retVal);
						else
							sprintf_s(buffer, sizeof(buffer), "  if (%s != 0) return %s;\n", ci(op1).c_str(), retVal);
					} else {
						if (instr->eBooleanTestType == INSTRUCTION_TEST_ZERO)
							sprintf_s(buffer, sizeof(buffer), "  if (%s == 0) return;\n", ci(op1).c_str());
						else
							sprintf_s(buffer, sizeof(buffer), "  if (%s != 0) return;\n", ci(op1).c_str());
					}
					appendOutput(buffer);
					break;

				// FarCry4 GeometryShader
				case OPCODE_EMIT_STREAM:
					sprintf_s(buffer, sizeof(buffer), "// Needs manual fix for instruction, maybe. \n//");
					appendOutput(buffer);
					ASMLineOut(c, pos, size);
					sprintf_s(buffer, sizeof(buffer), "m0.Append(0); \n");
					appendOutput(buffer);
					break;
				case OPCODE_CUT_STREAM:
					sprintf_s(buffer, sizeof(buffer), "// Needs manual fix for instruction, maybe. \n//");
					appendOutput(buffer);
					ASMLineOut(c, pos, size);
					sprintf_s(buffer, sizeof(buffer), "m0.RestartStrip(); \n");
					appendOutput(buffer);
					break;

				case OPCODE_NOP:
					// Used in MGSV:TPP, perhaps an artefact of having debug info enabled in shaders?
					break;

				default: logDecompileError("Unknown statement: " + string(statement)); return;
				}
				iNr++;
			}

			NextLine(c, pos, size);
			mLastStatement = instr;
		}

		// Moved this out of Opcode_ret, because it's possible to have more than one ret in a shader.
		// This is the last of a given shader, which seems more correct.
		// This fixes the double injection of "injectedScreenPos : SV_Position".
		WritePatches();
	}

// Restore the warning that was disabled outside of the main usage. sscanf_s warning on _int64.
#pragma warning(pop)

	void ParseCodeOnlyShaderType(Shader *shader, const char *c, size_t size) {
		mOutputRegisterValues.clear();
		mBooleanRegisters.clear();
		mCodeStartPos = mOutput.size();
		mDomainType = "tri";
		mInputControlPoints = 3;
		mOutputControlPoints = 3;
		mPartitioning = "fractional_odd";
		mOutputTopology = "triangle_cw";
		mMaxTessFactor = 64.0f;

		size_t pos = 0;
		while (pos < size) {
			// Ignore comments.
			if (!strncmp(c + pos, "//", 2)) {
				NextLine(c, pos, size);
				continue;
			}
			// Read statement.
			if (ReadStatement(c + pos) < 1) {
				return;
			}
			if (!strncmp(statement, "vs_", 3) || !strncmp(statement, "hs_", 3) || !strncmp(statement, "ds_", 3) ||
				!strncmp(statement, "gs_", 3) || !strncmp(statement, "ps_", 3) || !strncmp(statement, "cs_", 3)) {
				mShaderType = statement;
			}

			if (!strcmp(statement, "dcl_tessellator_domain")) {
				if (strstr(op1, "quad"))
					mDomainType = "quad";
				else if (strstr(op1, "tri"))
					mDomainType = "tri";
				else if (strstr(op1, "isoline"))
					mDomainType = "isoline";
			} else if (!strcmp(statement, "dcl_tessellator_partitioning")) {
				if (strstr(op1, "fractional_odd"))
					mPartitioning = "fractional_odd";
				else if (strstr(op1, "fractional_even"))
					mPartitioning = "fractional_even";
				else if (strstr(op1, "integer"))
					mPartitioning = "integer";
				else if (strstr(op1, "pow2"))
					mPartitioning = "pow2";
			} else if (!strcmp(statement, "dcl_tessellator_output_primitive")) {
				if (strstr(op1, "triangle_cw"))
					mOutputTopology = "triangle_cw";
				else if (strstr(op1, "triangle_ccw"))
					mOutputTopology = "triangle_ccw";
				else if (strstr(op1, "line"))
					mOutputTopology = "line";
				else if (strstr(op1, "point"))
					mOutputTopology = "point";
			} else if (!strcmp(statement, "dcl_input_control_point_count")) {
				mInputControlPoints = atoi(op1);
			} else if (!strcmp(statement, "dcl_output_control_point_count")) {
				mOutputControlPoints = atoi(op1);
			} else if (!strcmp(statement, "dcl_hs_max_tessfactor")) {
				sscanf_s(op1, "l(%f)", &mMaxTessFactor);
			} else if (!strcmp(statement, "dcl_thread_group")) {
				sscanf_s(statement, "dcl_thread_group %d, %d, %d", &mThreadGroup[0], &mThreadGroup[1], &mThreadGroup[2]);
			}
			NextLine(c, pos, size);
		}
	}

	// The StereoParams are nearly always useful, but the depth buffer texture is rarely used.
	// Adding .ini declaration, since declaring it doesn't cost anything and saves typing them in later.
	void WriteAddOnDeclarations() {
		string declaration = "\n\n// 3Dmigoto declarations\n";

		// Also inject the helper macro of 'cmp' to fix any boolean comparisons.
		// This is a bit of a hack, but simply adds a "-" in front of the comparison,
		// which negates the bool comparison from 1:0 to -1:0.
		//    r0.y = cmp(0 < r0.x);   becomes
		//    r0.y = -(0 < r0.x);
		// This allows us to avoid having helper routines, and needing different
		// variants for different swizzle sizes, like .xy or .xyz.
		declaration += "#define cmp -\n";

		if (IniParamsReg >= 0)
			declaration += "Texture1D<float4> IniParams : register(t" + std::to_string(IniParamsReg) + ");\n";
		if (StereoParamsReg >= 0)
			declaration += "Texture2D<float4> StereoParams : register(t" + std::to_string(StereoParamsReg) + ");\n";
		if (mZRepair_DepthBuffer)
			declaration += "Texture2D<float4> InjectedDepthTexture : register(t126);\n";
		declaration += "\n";
		mOutput.insert(mOutput.end(), declaration.c_str(), declaration.c_str() + declaration.length());
	}

	// Header for the file, version and time stamp. Skipping first line of file, to allow
	// it to be used for on-screen comments.
	void WriteHeaderDeclarations() {
		string header = "// ---- Created with 3Dmigoto v" + string(VER_FILE_VERSION_STR) + " on " + LogTime();
		// using .begin() to ensure first lines in files.
		mOutput.insert(mOutput.begin(), header.c_str(), header.c_str() + header.length());
	}

	// Assembles the final Hull Shader, structuring both the Patch Constant phase function
	// and the Control Point phase main entrypoint.
	void AssembleFinalShader() {
		bool isHS = (mShaderType.substr(0, 3) == "hs_");
		if (!isHS)
			return;

		vector<char> finalOut;

		string structs = mStructInput + mStructOutput + mStructPatchConstants;
		finalOut.insert(finalOut.end(), structs.begin(), structs.end());

		string oDecls = "  uint4 bitmask = 0, uiDest = 0;\n  float4 fDest = 0;\n";
		for (int k = 0; k < 32; k++) {
			oDecls += "  float4 o" + to_string(k) + " = 0;\n";
			oDecls += "  float4 p" + to_string(k) + " = 0;\n";
		}

		string forkHeader = "HS_PATCH_CONSTANTS PatchConstantFunc(\n  InputPatch<HS_INPUT_CP, " + to_string(mInputControlPoints) +
							"> ip,\n  uint PatchID : SV_PrimitiveID)\n{\n  HS_PATCH_CONSTANTS patchConsts = (HS_PATCH_CONSTANTS)0;\n" +
							oDecls + mHSUnpackCode;
		finalOut.insert(finalOut.end(), forkHeader.begin(), forkHeader.end());
		finalOut.insert(finalOut.end(), mHSForkPhase.begin(), mHSForkPhase.end());

		string forkPack = "\n";
		for (size_t k = 0; k < mHSPatchConstantAssignments.size(); k++) {
			forkPack += "  patchConsts." + mHSPatchConstantAssignments[k].targetField + " = o" +
						to_string(mHSPatchConstantAssignments[k].slot) + "." + mHSPatchConstantAssignments[k].mask + ";\n";
		}
		string forkFooter = forkPack + "  return patchConsts;\n}\n\n";
		finalOut.insert(finalOut.end(), forkFooter.begin(), forkFooter.end());

		string mainHeader =
			"[domain(\"" + mDomainType +
			"\")]\n"
			"[partitioning(\"" +
			mPartitioning +
			"\")]\n"
			"[outputtopology(\"" +
			mOutputTopology +
			"\")]\n"
			"[outputcontrolpoints(" +
			to_string(mOutputControlPoints) +
			")]\n"
			"[patchconstantfunc(\"PatchConstantFunc\")]\n"
			"HS_OUTPUT_CP main(\n  InputPatch<HS_INPUT_CP, " +
			to_string(mInputControlPoints) +
			"> ip,\n  uint i : SV_OutputControlPointID,\n  uint PatchID : SV_PrimitiveID)\n{\n  HS_OUTPUT_CP output = (HS_OUTPUT_CP)0;\n" +
			oDecls + mHSUnpackCode;
		finalOut.insert(finalOut.end(), mainHeader.begin(), mainHeader.end());
		finalOut.insert(finalOut.end(), mHSControlPointPhase.begin(), mHSControlPointPhase.end());

		string mainPack = "\n";
		for (size_t k = 0; k < mHSCPOutputAssignments.size(); k++) {
			mainPack += "  output." + mHSCPOutputAssignments[k].targetField + " = o" + to_string(mHSCPOutputAssignments[k].slot) + "." +
						mHSCPOutputAssignments[k].mask + ";\n";
		}
		string mainFooter = mainPack + "  return output;\n}\n";
		finalOut.insert(finalOut.end(), mainFooter.begin(), mainFooter.end());

		mOutput.insert(mOutput.end(), finalOut.begin(), finalOut.end());
	}
};

const string DecompileBinaryHLSL(ParseParameters &params, bool &patched, std::string &shaderModel, bool &errorOccurred) {
	Decompiler d;

	d.mCodeStartPos = 0;
	d.mCorrectedIndexRegisters.clear();
	d.mOutput.reserve(16 * 1024);
	d.mErrorOccurred = false;
	d.mShaderType = "unknown";

	// Copy parse parameters
	d.StereoParamsReg = params.StereoParamsReg;
	d.IniParamsReg = params.IniParamsReg;
	d.mFixSvPosition = params.fixSvPosition;
	d.mRecompileVs = params.recompileVs;
	d.mPatched = false;
	d.ZRepair_DepthTexture1 = params.ZRepair_DepthTexture1;
	d.ZRepair_DepthTexture2 = params.ZRepair_DepthTexture2;
	d.ZRepair_DepthTextureReg1 = params.ZRepair_DepthTextureReg1;
	d.ZRepair_DepthTextureReg2 = params.ZRepair_DepthTextureReg2;
	d.ZRepair_Dependencies1 = params.ZRepair_Dependencies1;
	d.ZRepair_Dependencies2 = params.ZRepair_Dependencies2;
	d.ZRepair_ZPosCalc1 = params.ZRepair_ZPosCalc1;
	d.ZRepair_ZPosCalc2 = params.ZRepair_ZPosCalc2;
	d.ZRepair_PositionTexture = params.ZRepair_PositionTexture;
	d.ZRepair_WorldPosCalc = params.ZRepair_WorldPosCalc;
	d.mZRepair_DepthBuffer = params.ZRepair_DepthBuffer;
	d.BackProject_Vector1 = params.BackProject_Vector1;
	d.BackProject_Vector2 = params.BackProject_Vector2;
	d.InvTransforms = params.InvTransforms;
	d.ObjectPos_ID1 = params.ObjectPos_ID1;
	d.ObjectPos_ID2 = params.ObjectPos_ID2;
	d.ObjectPos_MUL1 = params.ObjectPos_MUL1;
	d.ObjectPos_MUL2 = params.ObjectPos_MUL2;
	d.MatrixPos_ID1 = params.MatrixPos_ID1;
	d.MatrixPos_MUL1 = params.MatrixPos_MUL1;

	// Decompile binary.
	// This can crash, because of unknown or unseen syntax, so we wrap it in try/catch
	// block to handle any exceptions and mark the shader as presently bad.
	// In order for this to work, the /EHa option must be enabled for code-generation
	// so that system level exceptions are caught too.
	// It's worth noting that some fatal exceptions will still bypass this catch,
	// like a stack corruption, stack overflow, or out of memory, and crash the game.
	try {
		Shader *shader = DecodeDXBC((uint32_t *)params.bytecode);
		if (!shader)
			return string();

		// Scan shader type and tessellation attributes prior to signature generation
		d.ParseCodeOnlyShaderType(shader, params.decompiled, params.decompiledSize);
		d.ReadResourceBindings(params.decompiled, params.decompiledSize);
		d.ParseBufferDefinitions(shader, params.decompiled, params.decompiledSize);
		d.WriteResourceDefinitions();
		d.WriteAddOnDeclarations();

		d.ParseInputSignature(shader, params.decompiled, params.decompiledSize);
		d.ParsePatchConstantSignature(params.decompiled, params.decompiledSize);
		d.ParseOutputSignature(params.decompiled, params.decompiledSize);

		if (!params.ZeroOutput) {
			d.ParseCode(shader, params.decompiled, params.decompiledSize);
		} else {
			d.WriteZeroOutputSignature(params.decompiled, params.decompiledSize);
		}

		if (d.mShaderType.substr(0, 3) == "hs_") {
			d.AssembleFinalShader();
		} else {
			if (!params.ZeroOutput)
				d.mOutput.push_back('}');
		}

		d.WriteHeaderDeclarations();

		shaderModel = d.mShaderType;
		errorOccurred = d.mErrorOccurred;
		FreeShaderInfo(shader->sInfo);
		delete shader;
		patched = d.mPatched;
		return string(d.mOutput.begin(), d.mOutput.end());
	} catch (...) {
		// Fatal error, but catch it and mark it as bad.
		LogInfo("   ******* Exception caught while decompiling shader ******\n");
		errorOccurred = true;
		return string();
	}
}
