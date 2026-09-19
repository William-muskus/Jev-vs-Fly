These meshes are not shipped in git.

On the computer that already has the Cults STLs, from the repo root:

    pip install trimesh
    python -m game.wizard_assets --src "C:\Users\Willi\Downloads\harry-potter-chess20241003-1-ye3pe1\nbauchat\harry-potter-chess"

That writes k.glb q.glb b.glb n.glb r.glb p.glb here. Hard-reload the 3D hall
(Ctrl+Shift+R). The board sniffs these files directly; an empty manifest in git
must not hide them.

Until the GLBs exist on *this* machine, the engine builds original gothic-Staunton
stone. Cloud recordings never include the Cults sculpts.

