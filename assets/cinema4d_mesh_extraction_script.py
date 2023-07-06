import c4d
from c4d import gui
# Welcome to the world of Python
import cPickle

# Script state in the menu or the command palette
# Return True or c4d.CMD_ENABLED to enable, False or 0 to disable
# Alternatively return c4d.CMD_ENABLED|c4d.CMD_VALUE to enable and check/mark
#def state():
#    return True

# Main function
def main():
    doc = c4d.documents.GetActiveDocument()
    doc = doc.GetClone()
    obj = doc.SearchObject("face")
    print(obj)
    flags = c4d.MODELINGCOMMANDFLAGS_NONE
    bc = c4d.BaseContainer()
    mode = c4d.MODELINGCOMMANDMODE_ALL
    res = c4d.utils.SendModelingCommand(command=c4d.MCOMMAND_TRIANGULATE, list=[obj],
                                        mode=mode, bc=bc, doc=doc, flags=flags)
    c4d.EventAdd()
    weighttag = obj.GetTag(c4d.Tvertexmap, 0)
    weights = weighttag.GetAllHighlevelData()
    points = obj.GetAllPoints()
    points = [[p.x,p.y,p.z] for p in points]
    tris = obj.GetAllPolygons()
    tris = [[t.a, t.b, t.c] for t in tris]
    data = {
        'vertices' : points,
        'tris' : tris,
        'weights' : weights
    }
    with open("D:\\full_bfm_mesh_with_bg_v2.pkl", "wb") as f:
        cPickle.dump(data, f)
    print ("done!")


# Execute main()
if __name__=='__main__':
    main()