#include <Python.h>
#include <structmember.h>
#include "engine.h"
#include <numpy/arrayobject.h>
#include <pthread.h>

//#define Py_BEGIN_ALLOW_THREADS 
//#define Py_END_ALLOW_THREADS 

typedef struct {
	PyObject_HEAD
	PyObject *ep_capsule;
	PyObject *mutex_capsule;
} interop_EngineObject;

static pthread_mutex_t *
extract_mutex (interop_EngineObject *self)
{
	if (!self->mutex_capsule)
	{
		PyErr_SetString(PyExc_Exception, "mutex_capsule was NULL.");
		return NULL;
	}

	if (!PyCapsule_IsValid(self->mutex_capsule, "mutex"))
	{
		PyErr_SetString(PyExc_Exception, "mutex_capsule was not a capsule.");
		return NULL;
	}

	pthread_mutex_t * mutex = (pthread_mutex_t *)PyCapsule_GetPointer(self->mutex_capsule, "mutex");

	return mutex;
}

static const Engine *
extract_engine (interop_EngineObject *self)
{
	if (!self->ep_capsule)
	{
		PyErr_SetString(PyExc_Exception, "ep_capsule was NULL.");
		return NULL;
	}

	if (self->ep_capsule == Py_None)
	{
		PyErr_SetString(PyExc_Exception, "Engine is not yet open.");
		return NULL;
	}

	if (!PyCapsule_IsValid(self->ep_capsule, "engine"))
	{
		PyErr_SetString(PyExc_Exception, "ep_capsule was not a capsule.");
		return NULL;
	}

	const Engine * ep = (const Engine *)PyCapsule_GetPointer(self->ep_capsule, "engine");

	return ep;
}

static PyMemberDef interop_Engine_members[] = {
	{"ep_capsule", T_OBJECT_EX, offsetof(interop_EngineObject, ep_capsule), 0, "Engine pointer"},
	{"mutex_capsule", T_OBJECT_EX, offsetof(interop_EngineObject, mutex_capsule), 0, "pthread mutex"}
};

static PyObject *
interop_Engine_open (interop_EngineObject *self, PyObject *args) 
{
	if (self->ep_capsule != Py_None)
	{
		PyErr_SetString(PyExc_Exception, "Engine is already open.");
		return NULL;
	}

	pthread_mutex_t * mutex = extract_mutex (self);

	const char * startcmd = NULL;
	if (!PyArg_ParseTuple(args, "|z", &startcmd))
		return NULL;

	Engine * retval;
	Py_BEGIN_ALLOW_THREADS
	pthread_mutex_lock (mutex);
	retval = engOpen (startcmd);
	pthread_mutex_unlock (mutex);
	Py_END_ALLOW_THREADS

	if (!retval)
  {
		PyErr_SetString(PyExc_Exception, "Failed to open engine.");
		return NULL;
	}

	Py_DECREF(Py_None);
	self->ep_capsule = PyCapsule_New((void*)retval, "engine", NULL);

	if (!self->ep_capsule)
	{
		PyErr_SetString(PyExc_Exception, "Failed to initialize ep capsule.");
		return NULL;
	}

	assert (retval == extract_ep (self));

	printf("Opened engine\n");

	return Py_BuildValue("");
}

static PyObject *
interop_Engine_close (interop_EngineObject *self) 
{
	const Engine * ep = extract_engine (self);
	if (!ep)
		return NULL;

	pthread_mutex_t * mutex = extract_mutex (self);
	if (!mutex)
		return NULL;

	int retval;
	Py_BEGIN_ALLOW_THREADS
	pthread_mutex_lock (mutex);
	retval = engClose (ep);
	pthread_mutex_unlock (mutex);
	Py_END_ALLOW_THREADS

	if (retval)
	{
		PyErr_SetString(PyExc_Exception, "Failed to close engine.");
		return NULL;
	}

	Py_DECREF (self->ep_capsule);
	Py_INCREF (Py_None);
	self->ep_capsule = Py_None;

	return Py_BuildValue("");
}

static PyObject *
interop_Engine_getVariable (interop_EngineObject *self, PyObject *args) 
{
	const Engine * ep = extract_engine (self);
	if (!ep)
		return NULL;

	pthread_mutex_t * mutex = extract_mutex (self);
	if (!mutex)
		return NULL;

	const char * string;
	if (!PyArg_ParseTuple(args, "s", &string))
		return NULL;

	mxArray * retval;
	Py_BEGIN_ALLOW_THREADS
	pthread_mutex_lock (mutex);
	retval = engGetVariable (ep, string);
	pthread_mutex_unlock (mutex);
	Py_END_ALLOW_THREADS

	if (!retval)
	{
		PyErr_SetString(PyExc_Exception, "Failed to get variable.");
		return NULL;
	}

	mwSize nd = mxGetNumberOfDimensions(retval);
	npy_intp  pydims[NPY_MAXDIMS];
	const mwSize *dims = mxGetDimensions(retval);
	int i;
	for (i=0; i<nd; i++)  pydims[i] = (npy_intp)(dims[i]);
	bool c = mxIsComplex(retval);
	PyArrayObject * result = PyArray_SimpleNew(
														(npy_intp)(nd),
														pydims,
														c ? PyArray_CDOUBLE : PyArray_DOUBLE);
	if (!result)
	{
		PyErr_SetString(PyExc_Exception, "Failed to initialize array");
		mxDestroyArray (retval);
		return NULL;
	}
	mwSize lRows = mxGetM(retval);
	mwSize lCols = mxGetN(retval);
	const double * lPR = mxGetPr(retval);
	if (c) {
		const double * lPI = mxGetPi(retval);

		mwIndex lCol;	
		for (lCol = 0; lCol < lCols; lCol++) {
			unsigned char *lDst = result->data + result->strides[1]*lCol;
			mwIndex lRow;
			for (lRow = 0; lRow < lRows; lRow++, lDst += result->strides[0]) {
				((double*)lDst)[0] = *lPR++;
				((double*)lDst)[1] = *lPI++;
			}
		}
	} else {
		mwIndex lCol;
		for (lCol = 0; lCol < lCols; lCol++) {
			unsigned char *lDst = result->data + result->strides[1]*lCol;
			mwIndex lRow;
			for (lRow = 0; lRow < lRows; lRow++, lDst += result->strides[0]) {
				*(double*)lDst = *lPR++;
			}
		}
	}
	mxDestroyArray (retval);
	return PyArray_Return(result);
}

static PyObject *
interop_Engine_putVariable (interop_EngineObject *self, PyObject *args) 
{
	const Engine * ep = extract_engine (self);
	if (!ep)
		return NULL;

	pthread_mutex_t * mutex = extract_mutex (self);
	if (!mutex)
		return NULL;

	const char * string;
	const PyArrayObject * array;
	if (!PyArg_ParseTuple(args, "sO!", &string, &PyArray_Type, &array))
		return NULL;

	mwSize nd = array->nd;

	if (nd > 2)
	{
		PyErr_SetString(PyExc_Exception, "Only supports arrays of 1 or 2 dimensions.");
		return NULL;
	}

	int i;
	mwSize dims [NPY_MAXDIMS];
	for (i = 0; i < nd; ++i) dims[i] = array->dimensions[i];

	mxClassID classid;
	mxComplexity complexflag;
	switch (array->descr->kind)
	{
		case 'f':
			classid = mxDOUBLE_CLASS;
			complexflag = mxREAL;
			break;
		case 'c':
			classid = mxDOUBLE_CLASS;
			complexflag = mxCOMPLEX;
			break;
		case 'b':
		case 'i':
		case 'u':
		case 'S':
		case 'U':
		case 'V':
		default:
			PyErr_Format(PyExc_Exception, "Unsupported array type %c.", array->descr->kind);
			return NULL;
	}

	mxArray * marray = mxCreateNumericArray (nd, dims, classid, complexflag);

	if (!marray)
	{
		PyErr_SetString(PyExc_Exception, "Failed to initialize array.");
		return NULL;
	}

	mwSize lRows = mxGetM(marray);
	mwSize lCols = mxGetN(marray);
	double * lPR = mxGetPr(marray);
	if (complexflag == mxCOMPLEX) {
		double * lPI = mxGetPi(marray);

		mwIndex lCol;	
		for (lCol = 0; lCol < lCols; lCol++) {
			unsigned char *lDst = array->data + array->strides[1]*lCol;
			mwIndex lRow;
			for (lRow = 0; lRow < lRows; lRow++, lDst += array->strides[0]) {
				*lPR++ = ((double*)lDst)[0];
				*lPI++ = ((double*)lDst)[1];
			}
		}
	} else {
		mwIndex lCol;
		for (lCol = 0; lCol < lCols; lCol++) {
			unsigned char *lDst = array->data + array->strides[1]*lCol;
			mwIndex lRow;
			for (lRow = 0; lRow < lRows; lRow++, lDst += array->strides[0]) {
				*lPR++ = *(double*)lDst;
			}
		}
	}

	int retval;
	Py_BEGIN_ALLOW_THREADS
	pthread_mutex_lock (mutex);
	retval = engPutVariable (ep, string, marray);
	pthread_mutex_unlock (mutex);
	Py_END_ALLOW_THREADS

	if (retval)
	{
		PyErr_SetString(PyExc_Exception, "Failed to put variable.");
		return NULL;
	}

	return Py_BuildValue("");
}

static PyObject *
interop_Engine_evalString (interop_EngineObject *self, PyObject *args) 
{
	const Engine * ep = extract_engine (self);
	if (!ep)
		return NULL;

	pthread_mutex_t * mutex = extract_mutex (self);
	if (!mutex)
		return NULL;

	const char * string;
	if (!PyArg_ParseTuple(args, "s", &string))
		return NULL;

	int retval;
	Py_BEGIN_ALLOW_THREADS
	pthread_mutex_lock (mutex);
	retval = engEvalString (ep, string);
	pthread_mutex_unlock (mutex);
	Py_END_ALLOW_THREADS

	if (retval)
	{
		PyErr_SetString(PyExc_Exception, "Failed to eval string.");
		return NULL;
	}

	return Py_BuildValue("");
}

static PyObject *
interop_Engine_outputBuffer (interop_EngineObject *self, PyObject *args) 
{
	const Engine * ep = extract_engine (self);
	if (!ep)
		return NULL;

	pthread_mutex_t * mutex = extract_mutex (self);
	if (!mutex)
		return NULL;

	const char *p = NULL;
	int n = 0;

	if (!PyArg_ParseTuple(args, "|zi", &p, &n))
		return NULL;

	int retval;
	Py_BEGIN_ALLOW_THREADS
	pthread_mutex_lock (mutex);
	retval = engOutputBuffer (ep, p, n);
	pthread_mutex_unlock (mutex);
	Py_END_ALLOW_THREADS  

	if (retval)
  {
    PyErr_SetString(PyExc_Exception, "Failed to set output buffer.");
    return NULL;
  }

  return Py_BuildValue("");
}

static PyObject *
interop_Engine_openSingleUse (interop_EngineObject * self) 
{
	PyErr_SetString(PyExc_NotImplementedError, "interop.Engine.openSingleUse");
	return NULL;
}

static PyObject *
interop_Engine_getVisible (interop_EngineObject * self) 
{
	PyErr_SetString(PyExc_NotImplementedError, "interop.Engine.getVisible");
	return NULL;
}

static PyObject *
interop_Engine_setVisible (interop_EngineObject * self) 
{
	PyErr_SetString(PyExc_NotImplementedError, "interop.Engine.setVisible");
	return NULL;
}

static PyMethodDef interop_Engine_methods[] = {
	{"open", (PyCFunction)interop_Engine_open, METH_VARARGS, NULL},
	{"close", (PyCFunction)interop_Engine_close, METH_NOARGS, NULL},
	{"getVariable", (PyCFunction)interop_Engine_getVariable, METH_VARARGS, NULL},
	{"putVariable", (PyCFunction)interop_Engine_putVariable, METH_VARARGS, NULL},
	{"evalString", (PyCFunction)interop_Engine_evalString, METH_VARARGS, NULL},
	{"outputBuffer", (PyCFunction)interop_Engine_outputBuffer, METH_VARARGS, NULL},
	{"openSingleUse", (PyCFunction)interop_Engine_openSingleUse, METH_NOARGS, NULL},
	{"getVisible", (PyCFunction)interop_Engine_getVisible, METH_NOARGS, NULL},
	{"setVisible", (PyCFunction)interop_Engine_setVisible, METH_NOARGS, NULL},
	{NULL, NULL}
};

static void
interop_Engine_dealloc(interop_EngineObject* self)
{
	if (self->ep_capsule != Py_None)
		interop_Engine_close (self);
	pthread_mutex_t * mutex = extract_mutex (self);
	if (mutex)
		pthread_mutex_destroy (mutex);
	Py_XDECREF(self->ep_capsule);
	Py_XDECREF(self->mutex_capsule);
	Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
interop_Engine_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	interop_EngineObject *self;

	self = (interop_EngineObject*)type->tp_alloc(type, 0);
	if (self)
	{
		Py_INCREF(Py_None);
		self->ep_capsule = Py_None;
		pthread_mutex_t * mutex = malloc(sizeof(pthread_mutex_t));
		pthread_mutex_init (mutex, NULL);
		self->mutex_capsule = PyCapsule_New((void*)mutex, "mutex", NULL);
		if (!self->mutex_capsule)
		{
			Py_DECREF(self);
			return NULL;
		}
	}

	return (PyObject*)self;
}

static PyTypeObject interop_EngineType = {
	PyObject_HEAD_INIT(NULL)
	"interop.Engine",												/* tp_name */
	sizeof(interop_EngineObject),						/* tp_basicsize */
	0,																			/* tp_itemsize */
	(destructor)interop_Engine_dealloc,			/* tp_dealloc */
	0,																			/* tp_print */
	0,																			/* tp_getattr */
	0,																			/* tp_setattr */
	0,																			/* tp_reserved */
	0,																			/* tp_repr */
	0,																			/* tp_as_number */
	0,																			/* tp_as_sequence */
	0,																			/* tp_as_mapping */
	0,																			/* tp_hash  */
	0,																			/* tp_call */
	0,																			/* tp_str */
	0,																			/* tp_getattro */
	0,																			/* tp_setattro */
	0,																			/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,	/* tp_flags */
	"Engine objects", 											/* tp_docs */
	0,																			/* tp_traverse */
	0,																			/* tp_clear */
	0,																			/* tp_richcompare */
	0,																			/* tp_weaklistoffset */
	0,																			/* tp_iter */
	0,																			/* tp_iternext */
	interop_Engine_methods,									/* tp_methods */
	interop_Engine_members,									/* tp_members */
	0,																			/* tp_getset */
	0,																			/* tp_base */
	0,																			/* tp_dict */
	0,																			/* tp_descr_get */
	0,																			/* tp_descr_set */
	0,																			/* tp_dictoffset */
	0,																			/* tp_init */
	//(initproc)interop_Engine_init,					/* tp_init */
	0,																			/* tp_alloc */
	interop_Engine_new,											/* tp_new */
};	

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"interop",
	"Wraps the MATLAB C engine API.",
	-1,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit_interop(void)
{
	import_array();

	if (PyType_Ready(&interop_EngineType) < 0)
		return NULL;

	PyObject *m = PyModule_Create(&moduledef);
	if (!m)
		return NULL;

	Py_INCREF(&interop_EngineType);
	PyModule_AddObject(m, "Engine", (PyObject*)&interop_EngineType);

	return m;
}
