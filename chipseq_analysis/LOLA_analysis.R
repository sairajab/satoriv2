## ----echo=FALSE---------------------------------------------------------------
# These settings make the vignette prettier
knitr::opts_chunk$set(results="hold", message=FALSE)

## ----Load a regionDB----------------------------------------------------------

args <- commandArgs(TRUE)

library("LOLA")
dbPath = system.file("extdata", "tair10", package="LOLA")
print(dbPath)
regionDB = loadRegionDB(dbPath)

## ----Look at the elements of a regionDB---------------------------------------
names(regionDB)

## ----Load sample user sets and universe---------------------------------------
# data("sample_input", package="LOLA") # load userSets
# data("sample_universe", package="LOLA") # load userUniverse

queryA = readBed(args[1])#")
ocUniverse = readBed(args[2])
## ----Run the calculation------------------------------------------------------
locResults = runLOLA(queryA, ocUniverse, regionDB, cores=1)

## -----------------------------------------------------------------------------
colnames(locResults)
head(locResults)

## -----------------------------------------------------------------------------
locResults[order(support, decreasing=TRUE),]

## -----------------------------------------------------------------------------
locResults[order(maxRnk, decreasing=TRUE),]

## ----Write results------------------------------------------------------------
writeCombinedEnrichment(locResults, outFolder= args[3])

## ----Write split results------------------------------------------------------
#writeCombinedEnrichment(locResults, outFolder= "lolaResults", includeSplits=TRUE)

## ----Extracting overlaps------------------------------------------------------
#oneResult = locResults[2,]
#extractEnrichmentOverlaps(oneResult, userSets, regionDB)

## ----Grabbing individual region sets------------------------------------------
#getRegionSet(regionDB, collections="interacting_tfs", filenames="C2C2dof_tnt_AT1G64620_colamp.target.all.bed")

## ----Grabbing individual region sets from disk--------------------------------
#getRegionSet(dbPath, collections="interacting_tfs", filenames="C2C2dof_tnt_AT1G64620_colamp.target.all.bed")

