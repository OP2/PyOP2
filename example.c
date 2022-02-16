#include <petsc.h>

static char help[] = "Compare the speeds of different packing schemes for unstructures meshes.\n";

const static char MYEVENTNAME[] = "MyEventName";

/* Force some tensor stuff */
const static PetscInt TENSOR_SHAPE = 2;
const static PetscInt NEDGEPOINTS = 3;
const static PetscInt NVERTPOINTS = 1;

const static PetscInt NCELLDOFS = TENSOR_SHAPE;
const static PetscInt NEDGEDOFS = NEDGEPOINTS*TENSOR_SHAPE;
const static PetscInt NVERTDOFS = NVERTPOINTS*TENSOR_SHAPE;
const static PetscInt NEDGESPERCELL = 3;
const static PetscInt NVERTSPERCELL = 3;

PetscErrorCode DMVecPack_PetscUnmemoized(const DM dm,const Vec vec,const PetscInt nrepeats)
{
  PetscScalar    *values = NULL;
  PetscInt       cStart,cEnd,csize,r,i,c;
  PetscLogEvent	 event;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(MYEVENTNAME,0,&event);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  for (r=0; r<nrepeats;++r)
  {
    for (c=cStart; c<cEnd; ++c)
    {
      // PetscAttachDebugger();
      DMPlexVecGetClosure(dm, NULL, vec, c, &csize, &values);
      /* Touch the data */
      for (i=0; i<csize; i++) {
	values[i]++;
      }
      DMPlexVecSetClosure(dm, NULL, vec, c, values, INSERT_VALUES);
      DMPlexVecRestoreClosure(dm, NULL, vec, c, &csize, &values);
    }
  }
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMVecPack_PetscMemoized(const DM dm,const Vec vec,const PetscInt nrepeats)
{
  PetscScalar	 *data = NULL;
  const PetscInt *indices = NULL;
  PetscInt    	 cStart,cEnd,r,c,i,j,cloff,cldof,offset,dof;
  PetscErrorCode ierr;
  PetscSection	 section,clSection;
  IS		 clPoints;
  PetscLogEvent	 event;

  PetscFunctionBegin;
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetClosureIndex(section,(PetscObject) dm,&clSection,&clPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(clPoints,&indices);CHKERRQ(ierr);
  ierr = VecGetArray(vec, &data);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  ierr = PetscLogEventRegister(MYEVENTNAME,0,&event);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  for (r=0; r<nrepeats;++r)
  {
    for (c=cStart; c<cEnd; ++c)
    {
      ierr = PetscSectionGetDof(clSection,c,&cldof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(clSection,c,&cloff);CHKERRQ(ierr);
      for (i=0; i<cldof; i+=2) {
	PetscSectionGetDof(section,indices[cloff+i],&dof);
	PetscSectionGetOffset(section,indices[cloff+i],&offset);
	for (j=0; j<dof; ++j) {
	  data[offset+j]++;
	}
      }
    }
  }
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  ierr = ISRestoreIndices(clPoints,&indices);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode Compute_Hardcoded(const DM dm,const Vec vec,const PetscInt nrepeats)
{
  PetscScalar	 *data = NULL;
  const PetscInt *indices = NULL;
  PetscInt    	 cStart,cEnd,r,c,cloff,offset,d;
  PetscErrorCode ierr;
  PetscSection	 section,clSection;
  IS		 clPoints;
  PetscLogEvent	 event;

  PetscFunctionBegin;
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetClosureIndex(section,(PetscObject) dm,&clSection,&clPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(clPoints,&indices);CHKERRQ(ierr);
  ierr = VecGetArray(vec, &data);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  /* Compose maps so that we do not need to call PetscSectionGetOffset twice */
  /* Inspired by DMPlexCreateClosureIndex */
  PetscSection	offsetSection;
  IS		offsets;
  PetscInt      *offsetPoints;
  PetscInt      offsetSize,myoff;

  /* First create the section */
  PetscSectionCreate(PETSC_COMM_WORLD,&offsetSection);
  PetscSectionSetChart(offsetSection,cStart,cEnd);
  for (c=cStart; c<cEnd; ++c) {
    PetscSectionSetDof(offsetSection,c,1+NEDGESPERCELL+NVERTSPERCELL);
  }
  PetscSectionSetUp(offsetSection);

  /* Then populate the IS */
  PetscSectionGetStorageSize(offsetSection,&offsetSize);
  PetscMalloc1(offsetSize,&offsetPoints);
  for (c=cStart; c<cEnd; ++c) {
    ierr = PetscSectionGetOffset(clSection,c,&cloff);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(offsetSection,c,&myoff);CHKERRQ(ierr);

    /* cell */
    PetscSectionGetOffset(section,indices[cloff],&offset);
    offsetPoints[myoff] = offset;

    /* edges */
    for (PetscInt e=0; e<NEDGESPERCELL; ++e) {
      PetscSectionGetOffset(section,indices[cloff+2*e+2],&offset);
      offsetPoints[myoff+e+1] = offset;
    }

    /* vertices */
    for (PetscInt v=0; v<NVERTSPERCELL; ++v) {
      PetscSectionGetOffset(section,indices[cloff+2*v+2*NEDGESPERCELL+2],&offset);
      offsetPoints[myoff+v+NEDGESPERCELL+1] = offset;
    }
  }
  ISCreateGeneral(PETSC_COMM_WORLD,offsetSize,offsetPoints,PETSC_OWN_POINTER,&offsets);

  const PetscInt *offsetIndices = NULL;
  ierr = ISGetIndices(offsets,&offsetIndices);CHKERRQ(ierr);

  const int stepsize = 1 + NEDGESPERCELL + NVERTSPERCELL;

  ierr = PetscLogEventRegister(MYEVENTNAME,0,&event);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  for (r=0; r<nrepeats;++r)
  {
    for (c=cStart; c<cEnd; ++c)
    {
      offset = c * stepsize;

      /* cell */
      for (d=0; d<NCELLDOFS; d++) {
	data[offsetIndices[offset]+d]++;
      }

      /* edges */
      for (PetscInt e=0; e<NEDGESPERCELL; ++e) {
	for (d=0; d<NEDGEDOFS; d++) {
	  data[offsetIndices[offset+e+1]+d]++;
	}
      }

      /* vertices */
      for (PetscInt v=0; v<NVERTSPERCELL; ++v) {
	for (d=0; d<NVERTDOFS; d++) {
	  data[offsetIndices[offset+v+NEDGESPERCELL+1]+d]++;
	}
      }
    }
  }
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  ierr = ISRestoreIndices(clPoints,&indices);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode Compute_HardcodedLegacy(const DM dm,const Vec vec,const PetscInt nrepeats)
{
  PetscScalar	 *data = NULL;
  const PetscInt *indices = NULL;
  PetscInt    	 cStart,cEnd,r,c,cloff,offset,d;
  PetscErrorCode ierr;
  PetscSection	 section,clSection;
  IS		 clPoints;
  PetscLogEvent	 event;

  PetscFunctionBegin;
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetClosureIndex(section,(PetscObject) dm,&clSection,&clPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(clPoints,&indices);CHKERRQ(ierr);
  ierr = VecGetArray(vec, &data);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  const PetscInt arity = 1 + NEDGESPERCELL*NEDGEPOINTS + NVERTSPERCELL*NVERTPOINTS;

  /* Compose maps so that we do not need to call PetscSectionGetOffset twice */
  /* Inspired by DMPlexCreateClosureIndex */
  PetscSection	offsetSection;
  IS		offsets;
  PetscInt      *offsetPoints;
  PetscInt      offsetSize,myoff;

  /* First create the section */
  PetscSectionCreate(PETSC_COMM_WORLD,&offsetSection);
  PetscSectionSetChart(offsetSection,cStart,cEnd);
  for (c=cStart; c<cEnd; ++c) {
    PetscSectionSetDof(offsetSection,c,arity);
  }
  PetscSectionSetUp(offsetSection);

  /* Then populate the IS */
  PetscSectionGetStorageSize(offsetSection,&offsetSize);
  PetscMalloc1(offsetSize,&offsetPoints);
  for (c=cStart; c<cEnd; ++c) {
    ierr = PetscSectionGetOffset(clSection,c,&cloff);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(offsetSection,c,&myoff);CHKERRQ(ierr);

    /* cell */
    PetscSectionGetOffset(section,indices[cloff],&offset);
    offsetPoints[myoff] = offset;

    /* edges */
    for (PetscInt e=0; e<NEDGESPERCELL; ++e) {
      PetscSectionGetOffset(section,indices[cloff+2*e+2],&offset);
      for (PetscInt e_=0; e_<NEDGEPOINTS; ++e_) {
	offsetPoints[myoff+e*NEDGEPOINTS+1+e_] = offset+e_*TENSOR_SHAPE;
      }
    }

    /* vertices */
    for (PetscInt v=0; v<NVERTSPERCELL; ++v) {
      PetscSectionGetOffset(section,indices[cloff+2*v+2*NEDGESPERCELL+2],&offset);
      for (PetscInt v_=0; v_<NVERTPOINTS; ++v_) {
	offsetPoints[myoff+v*NVERTPOINTS+NEDGESPERCELL*NEDGEPOINTS+1+v_] = offset+v_*TENSOR_SHAPE;
      }
    }
  }
  ISCreateGeneral(PETSC_COMM_WORLD,offsetSize,offsetPoints,PETSC_OWN_POINTER,&offsets);

  const PetscInt *offsetIndices = NULL;
  ierr = ISGetIndices(offsets,&offsetIndices);CHKERRQ(ierr);

  ierr = PetscLogEventRegister(MYEVENTNAME,0,&event);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  for (r=0; r<nrepeats;++r)
  {
    for (c=cStart; c<cEnd; ++c)
    {
      for (PetscInt a=0; a<arity; ++a) {
	for (PetscInt d=0; d<TENSOR_SHAPE; ++d) {
	  data[offsetIndices[c*arity+a]+d]++;
	}
      }
    }
  }
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  ierr = ISRestoreIndices(clPoints,&indices);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CheckValues(const DM dm,const Vec vec,const PetscInt nrepeats)
{
  PetscInt          start,end,i,j,offset,dof;
  const PetscScalar *data=NULL;
  PetscSection      s;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm,&s);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vec,&data);CHKERRQ(ierr);

  /* Every cell should have been visited nrepeats times */
  ierr = DMPlexGetHeightStratum(dm, 0, &start, &end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    ierr = PetscSectionGetOffset(s,i,&offset);
    ierr = PetscSectionGetDof(s,i,&dof);
    for (j=0; j<dof; j++) {
      if (data[offset+j] != nrepeats) {
	SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Cell checks failed");
      }
    }
  }
  /* An edge will always be incident on one or two cells */
  ierr = DMPlexGetHeightStratum(dm, 1, &start, &end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    ierr = PetscSectionGetOffset(s,i,&offset);
    ierr = PetscSectionGetDof(s,i,&dof);
    for (j=0; j<dof; j++) {
      if (data[offset+j] != nrepeats && data[offset+j] != 2*nrepeats) {
	SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Edge checks failed");
      }
    }
  }
  ierr = VecRestoreArrayRead(vec,&data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{  
  DM	      dm;
  Vec 	      vec;
  PetscSection sec;
  PetscViewer viewer;
  MPI_Comm    comm;
  PetscErrorCode ierr;
  PetscInt    nrepeats, mode;
  PetscBool set;
  char         filename[20];

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  /* Parse user-provided options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-nrepeats",&nrepeats,&set);CHKERRQ(ierr);
  if (!set) SETERRQ(comm,PETSC_ERR_USER_INPUT,"Need to provide -nrepeats");
  ierr = PetscOptionsGetInt(NULL,NULL,"-mode",&mode,&set);CHKERRQ(ierr);
  if (!set) SETERRQ(comm,PETSC_ERR_USER_INPUT,"Need to provide -mode");
  ierr = PetscOptionsGetString(NULL,NULL,"-filename",filename,20,&set);CHKERRQ(ierr);
  if (!set) SETERRQ(comm,PETSC_ERR_USER_INPUT,"Need to provide -filename");

  /* Load the mesh */
  ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,filename);CHKERRQ(ierr);
  ierr = DMPlexCreateGmsh(comm,viewer,PETSC_TRUE,&dm);CHKERRQ(ierr);

  /* Create a vector */
  int c,e,v,cStart,cEnd,eStart,eEnd,vStart,vEnd,pStart,pEnd;

  DMPlexGetChart(dm, &pStart, &pEnd);
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);  /* cells */
  DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd);  /* edges */
  DMPlexGetHeightStratum(dm, 2, &vStart, &vEnd);  /* vertices */
  PetscSectionCreate(comm, &sec);
  PetscSectionSetChart(sec, pStart, pEnd);
  for(c=cStart;c<cEnd;++c)
      PetscSectionSetDof(sec,c,NCELLDOFS);
  for(v=vStart;v<vEnd;++v)
      PetscSectionSetDof(sec,v,NVERTDOFS);
  for(e = eStart; e < eEnd; ++e)
      PetscSectionSetDof(sec,e,NEDGEDOFS);
  PetscSectionSetUp(sec);

  DMSetLocalSection(dm, sec);
  DMGetLocalVector(dm, &vec);
  VecZeroEntries(vec);

  switch (mode)
  {
    case 0:
      ierr = DMVecPack_PetscUnmemoized(dm,vec,nrepeats);CHKERRQ(ierr);
      break;
    case 1:
      ierr = DMVecPack_PetscMemoized(dm,vec,nrepeats);CHKERRQ(ierr);
      break;
    case 2:
      ierr = Compute_Hardcoded(dm,vec,nrepeats);CHKERRQ(ierr);
      break;
    case 3:
      ierr = Compute_HardcodedLegacy(dm,vec,nrepeats);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(comm,PETSC_ERR_USER_INPUT,"Invalid mode specified");
  }

  ierr = CheckValues(dm, vec, nrepeats);

  /* Final cleanup */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
