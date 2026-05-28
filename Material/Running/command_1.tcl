*begin "version 2019.1.0.20  6-22-2021  9:42:20"
*menufilterset "*"
*menufilterdisable 
*createmark collections 1
*clearmark collections 1
*createmark collections 2
*clearmark collections 2
*createmark controllers 1
*clearmark controllers 1
*clearmarkall 3
*setelementcolormode 16
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*ME_CoreBehaviorAdjust "allowable_actions_policy=TC_lite"
*loaddefaultattributevaluesfromxml 
*templatefileset "D:/Hyperworks2019/templates/feoutput/abaqus/explicit"
*menufilterset "*"
*menufilterdisable 
*createmark collections 1
*clearmark collections 1
*createmark collections 2
*clearmark collections 2
*createmark controllers 1
*clearmark controllers 1
*clearmarkall 3
*setelementcolormode 16
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*settopologydisplaymode 0
*ME_CoreBehaviorAdjust "allowable_actions_policy=TC_lite"
*createmark sets 1
*clearmark sets 1
*createmark sets 1
*clearmark sets 1
*elementtype 60 9
*elementtype 104 7
*elementtype 103 7
*elementtype 206 1
*elementtype 5 9
*elementtype 55 9
*elementtype 205 7
*elementtype 208 7
*elementtype 56 2
*loaddefaultattributevaluesfromxml 
*createstringarray 19 "Abaqus " "Explicit " "ALESMOOTHINGS_DISPLAY_SKIP " \
  "EXTRANODES_DISPLAY_SKIP " "ACCELEROMETERS_DISPLAY_SKIP " "LOADCOLS_DISPLAY_SKIP " \
  "RETRACTORS_DISPLAY_SKIP " "VECTORCOLS_DISPLAY_SKIP " "SYSTCOLS_DISPLAY_SKIP " \
  "PRIMITIVES_DISPLAY_SKIP " "BLOCKS_DISPLAY_SKIP " "ELEMENTCLUSTERS_DISPLAY_SKIP " \
  "CROSSSECTION_DISPLAY_SKIP " "CONSTRAINEDRIGIDBODY_DISPLAY_SKIP " "RIGIDWALLS_DISPLAY_SKIP " \
  "SLIPRINGS_DISPLAY_SKIP " "CONTACTSURF_DISPLAY_SKIP " "IDRULES_SKIP" "IMPORT_MATERIAL_METADATA"
*feinputwithdata2 "\#stl\\stl" "1.stl" 0 0 0 0 0 1 19 1 0
*drawlistresetstyle 
*createmark elements 1 "all"
*createmark elements 2
*shrinkwrapmesh elements 1 2 0.2 30 12 1 0.3 0 0 0 0
*createmark elements 1 "all"
*clearmark elements 1
*createmark systems 1
*clearmark systems 1
*retainmarkselections 0
*createmark elements 1
*clearmark elements 1
*createmark systems 1
*clearmark systems 1
*retainmarkselections 0
*createstringarray 3 "HMBOMCOMMENTS_XML" "HMMATCOMMENTS_XML" "EXPORTIDS_SKIP"

set outFile "adaqusModel1.inp"
if { [file exists $outFile] } {
    file delete -force $outFile
}
*feoutputwithdata "D:/Hyperworks2019/templates/feoutput/abaqus/explicit" "adaqusModel1.inp" 0 0 1 1 3
hm_answernext yes
*deletemodel
# Session ended at "8-4-2021  9:42:02"
# return;
*quit 1;
