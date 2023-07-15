import c4d
from c4d import gui
import cPickle


def getTagsByName(obj):
    return { t.GetName():t for t in obj.GetTags() }


def triangulate_and_get_geom(doc, obj):
    flags = c4d.MODELINGCOMMANDFLAGS_NONE
    bc = c4d.BaseContainer()
    mode = c4d.MODELINGCOMMANDMODE_ALL
    c4d.utils.SendModelingCommand(command=c4d.MCOMMAND_TRIANGULATE, list=[obj],
                                        mode=mode, bc=bc, doc=doc, flags=flags)
    c4d.EventAdd()
    points = obj.GetAllPoints()
    points = [[p.x,p.y,p.z] for p in points]
    tris = obj.GetAllPolygons()
    tris = [[t.a, t.b, t.c] for t in tris]
    return points, tris


def main():
    doc = c4d.documents.GetActiveDocument()
    doc = doc.GetClone()
    obj = doc.SearchObject("face")

    points, tris = triangulate_and_get_geom(doc, obj)
    tags = getTagsByName(obj)
    #weights = tags['faceweight'].GetAllHighlevelData()
    mask_mouth_lower = tags['mouth_lower'].GetBaseSelect().GetAll(len(points))
    mask_mouth_upper = tags['mouth_upper'].GetBaseSelect().GetAll(len(points))

    obj = doc.SearchObject("teeth")
    teeth_points, teeth_tris = triangulate_and_get_geom(doc, obj)

    obj = doc.SearchObject("mouth")
    tags = getTagsByName(obj)
    mouth_points, mouth_tris = triangulate_and_get_geom(doc, obj)
    mouth_shadowmap = tags['shadowmap'].GetAllHighlevelData()

    data = {
        'vertices' : points,
        'tris' : tris,
        #'weights' : weights,
        'mask_mouth_lower' : mask_mouth_lower,
        'mask_mouth_upper' : mask_mouth_upper,
        'teeth_points' : teeth_points,
        'teeth_tris' : teeth_tris,
        'mouth_points' : mouth_points,
        'mouth_tris' : mouth_tris,
        'mouth_shadowmap' : mouth_shadowmap,
    }
    with open("D:\\full_bfm_mesh_with_bg_v6.pkl", "wb") as f:
        cPickle.dump(data, f)

    print ("done!")


# Execute main()
if __name__=='__main__':
    main()