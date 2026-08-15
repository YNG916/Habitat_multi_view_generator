# Robot proxy asset attribution

The finished robot-vacuum geometry and original albedo in `source/` are based on:

- **“Robot vacuum cleaner low poly”** by **Moryak** (`_moryak_`)
- Original model: https://sketchfab.com/3d-models/7230d8d80e8b4a82b4a34a5e7926d0d3
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License text: https://creativecommons.org/licenses/by/4.0/
- Retrieved from the attributed redistribution in:
  https://github.com/viliger2/RoR2_Roomba
- Source FBX SHA-256:
  `069f80f5023dc2e86eb95bae35b728bec4b42cc3ffea9d3c8b6d0485aaa36d9f`
- Source albedo SHA-256:
  `ae8c16c18def8589ecafd1846768db1fd07f1335847cc88210ecccde4dee6e15`

Changes made by this repository:

- converted the FBX coordinate system to Habitat Y-up;
- uniformly scaled it to a 0.46 m maximum horizontal diameter;
- centered X/Z and translated the authored bottom to local Y=0;
- produced red, green, and blue albedo variants for agent identity;
- exported the unchanged authored triangle topology to OBJ.

No procedural geometry is added to the finished robot body.
