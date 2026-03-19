
.. include:: macros.txt

Generating Source Code
----------------------

model_list
~~~~~~~~~~

This command prints a list of all models which have been defined up to this point in the execution of |symdiff|.  This is shown in the ``modellist1.py`` example:

.. code::

  symdiff('declare_model(x)')
  symdiff('declare_model(y)')
  l = model_list()
  for i in l:
    print('%s' % i)

The resulting output is then:

.. code::

  x
  y

ordered_list
~~~~~~~~~~~~

This command takes a list of 1 or more model names.  The resulting list is in the order necessary to ensure that the models are evaluated in the correct order.  In this example, ``ordered.py``, we define 2 models, and |symdiff| prints what order the models would have to be defined.

.. code::

  symdiff('define_model(b, a)')
  symdiff('define_model(d, b * c)')
  mylist = ordered_list('d')
  for i in mylist:
    print('%s' % i)

The resulting output is then:

.. code::

  b
  d

For ordering multiple model names, pass multiple names, or a list using this syntax.

.. code::

  ordered_list('a', 'b')
  args = ('a', 'b')
  ordered_list(*args)


remove_zeros
~~~~~~~~~~~~

This command removes all models whose evaluation results in ``0``.  Any models which rely on the definition of models will substitute a ``0`` in their expression for this model.  This is shown in the ``remove1.py`` example.

.. code::

  symdiff('define_model(x, 0)')
  symdiff('define_model(y, x + z)')
  remove_zeros()
  print('%s' % symdiff('model_value(y)'))

The resulting output is then:

.. code::

  z


subexpression
~~~~~~~~~~~~~

This command will evaluate all of the currently defined models and find common sub expressions.  If more than one dependent model uses the same sub-expression, |symdiff| will automatically substitute it with a new model with a generated name, as shown in the ``subexpression1.py`` example.

.. code::

  symdiff('define_model(x, simplify(y * z))')
  symdiff('define_model(z, simplify(z * y))')
  subexpression()
  l = model_list()
  for i in l:
    print("%s, %s" % (i, symdiff('model_value(%s)' % i)))

The resulting output is then:

.. code::

  unique0, (y * z)
  x, unique0
  z, unique0

The use of the ``simplify`` method is important to ensure that the subexpression elimination algorithm can recognize the common expressions.


