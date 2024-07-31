library(lidR) ; library(sf)
parentDir <- readClipboard()
setwd(parentDir)

# Get list of LAZ files that don't have a building in it
LAZFiles <- list.files(pattern="*.laz")
for(i in LAZFiles){
  tempLAZ <- readLAS(i)
  if(sum(tempLAZ$Classification==6)==0){
    print(i)
  }
}

# Using an AOI containing a non-building classified as a building, reclassify the LAZ
aoi <- read_sf("---_areaToReclassify.shp")
laz <- readLAS("C:\\---\\AC70.laz")
st_crs(laz) <- st_crs(aoi) # Copy PRJ

poi <- ~Classification == LASBUILDING # Get the specific LAZ classification that you want to change
clippedLAZ <- classify_poi(laz,LASLOWVEGETATION,poi=poi,roi=aoi) # Change the LAZ classification
writeLAS(clippedLAZ,"C:\\---\\AC70_out.laz")
