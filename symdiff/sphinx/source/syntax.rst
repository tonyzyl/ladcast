
.. include:: macros.txt

Syntax
------

Variables and Numbers
~~~~~~~~~~~~~~~~~~~~~

Variables and numbers are the basic building blocks for expressions.  A variable is defined as any sequence of characters not beginning with a number or underscore, (``_``), followed by any number of characters.  Note that the letters are case sensitive so that ``a`` and ``A`` are not the same variable.  Any other characters are considered to be either mathematical operators or invalid, even if there is no space between the character and the rest of the variable name.

|symdiff| supports |unicode| character sets, so that special characters such as ψ can be used.  Note however this feature should be used carefully when generating source code since these characters are not valid input for many computer languages.  Please see :ref:`sec__unicode` for more information.

Examples of valid variable names are:

* ``a, dog, var1, var_2``

Numbers can be integer or floating point.  Scientific notation is accepted as a valid syntax.  For example:

* ``1.0, 1.0e-2, 3.4E-4``

Basic Expressions
~~~~~~~~~~~~~~~~~

.. This code taken from devsim rst files for symdiff

.. _ch1__operators:

.. table:: Basic expressions involving unary, binary, and logical operators.

  +-------------------+--------------------------------------+
  | Expression        |   Description                        |
  +===================+======================================+
  | ``(exp1)``        |   Parenthesis for changing precedence|
  +-------------------+--------------------------------------+
  +-------------------+--------------------------------------+
  | ``+exp1``         |   Unary Plus                         |
  +-------------------+--------------------------------------+
  | ``-exp1``         |   Unary Minus                        |
  +-------------------+--------------------------------------+
  | ``!exp1``         |   Logical Not                        |
  +-------------------+--------------------------------------+
  +-------------------+--------------------------------------+
  | ``exp1 ^ exp2``   |   Exponentiation                     |
  +-------------------+--------------------------------------+
  +-------------------+--------------------------------------+
  | ``exp1 * exp2``   |   Multiplication                     |
  +-------------------+--------------------------------------+
  | ``exp1 / exp2``   |   Division                           |
  +-------------------+--------------------------------------+
  +-------------------+--------------------------------------+
  | ``exp1 + exp2``   |   Addition                           |
  +-------------------+--------------------------------------+
  | ``exp1 - exp2``   |   Subtraction                        |
  +-------------------+--------------------------------------+
  +-------------------+--------------------------------------+
  | ``exp1 < exp2``   |   Test Less                          |
  +-------------------+--------------------------------------+
  | ``exp1 <= exp2``  |   Test Less Equal                    |
  +-------------------+--------------------------------------+
  | ``exp1 > exp2``   |   Test Greater                       |
  +-------------------+--------------------------------------+
  | ``exp1 >= exp2``  |   Test Greater Equal                 |
  +-------------------+--------------------------------------+
  +-------------------+--------------------------------------+
  | ``exp1 == exp2``  |   Test Equality                      |
  +-------------------+--------------------------------------+
  | ``exp1 != exp2``  |   Test Inequality                    |
  +-------------------+--------------------------------------+
  +-------------------+--------------------------------------+
  | ``exp1 && exp2``  |   Logical And                        |
  +-------------------+--------------------------------------+
  +-------------------+--------------------------------------+
  | ``exp1 || exp2``  |   Logical Or                         |
  +-------------------+--------------------------------------+
  +-------------------+--------------------------------------+
  | ``variable``      |   Independent Variable               |
  +-------------------+--------------------------------------+
  | ``number``        |   Integer or decimal number          |
  +-------------------+--------------------------------------+

In :numref:`ch1__operators`, the basic syntax for the language is presented.  An expression may be composed
of variables and numbers joined with mathematical operators.  Order of operations is from bottom to top in order of increasing precedence.  Operators with the same level of precedence are contained within horizontal lines.

 In the expression ``a + b * c``, the multiplication will be performed before the addition.  In order to override this precedence, parenthesis are used.  For example, in ``(a + b) * c``, the addition operation is performed before the multiplication.

Functions
^^^^^^^^^


.. _ch1__functions:

.. table:: Predefined Functions.

  +---------------------------------------+---------------------------------------------------------------+
  | Function                              |   Description                                                 |
  +=======================================+===============================================================+
  | ``exp(exp1)``                         | exponent                                                      |
  +---------------------------------------+---------------------------------------------------------------+
  | ``log(exp1)``                         | natural log                                                   |
  +---------------------------------------+---------------------------------------------------------------+
  | ``pow(exp1, exp2)``                   | take exp1 to the power of exp2                                |
  +---------------------------------------+---------------------------------------------------------------+
  | ``ifelse(test, exp1, exp2)``          | if test is true, then evaluate exp1, otherwise exp2           |
  +---------------------------------------+---------------------------------------------------------------+
  | ``if(test, exp)``                     | if test is true, then evaluate exp, otherwise 0               |
  +---------------------------------------+---------------------------------------------------------------+


In :numref:`ch1__functions` are the built in functions of |symdiff|.  Note that the ``pow`` function uses the ``,`` operator to separate arguments.  In addition an expression like ``pow(a,b+y)`` is equivalent to an expression like ``a^(b+y)``.  Both ``exp`` and ``log`` are provided since many derivative expressions can be expressed in terms of these two functions.  It is possible to nest expressions within functions and vice-versa.

Special care should be used when using the exponentiation operator, since the unary minus has a higher precedence than the exponentiation operator.

.. code::

  >>> symdiff('x^-1 + 3')
  '(pow(x,(-1)) + 3)'
  >>> symdiff('-x^3')
  'pow((-x),3)'

Parenthesis are recommended to avoid ambiguity in this situation.

Commands
~~~~~~~~

.. _ch1__commands:

.. table:: Commands.

  +---------------------------------------+---------------------------------------------------------------+
  | Command                               |   Description                                                 |
  +=======================================+===============================================================+
  | ``diff(obj1, var)``                   | Take derivative  of ``obj1`` with respect to variable ``var`` |
  +---------------------------------------+---------------------------------------------------------------+
  | ``expand(obj)``	                  | Expand out all multiplications into a sum of products         |
  +---------------------------------------+---------------------------------------------------------------+
  | ``scale(obj)``                        | Get constant factor                                           |
  +---------------------------------------+---------------------------------------------------------------+
  | ``sign(obj)``                         | Get sign as ``1`` or ``-1``                                   |
  +---------------------------------------+---------------------------------------------------------------+
  | ``simplify(obj)``                     | Simplify as much as possible                                  |
  +---------------------------------------+---------------------------------------------------------------+
  | ``subst(obj1,obj2,obj3)``             | substitute ``obj3`` for ``obj2`` into ``obj1``                |
  +---------------------------------------+---------------------------------------------------------------+
  | ``unscaledval(obj)``                  | Get value without constant scaling                            |
  +---------------------------------------+---------------------------------------------------------------+
  | ``unsignedval(obj)``                  | Get unsigned value                                            |
  +---------------------------------------+---------------------------------------------------------------+


Commands are shown in :numref:`ch1__commands`.  While they appear to have the same form as functions, they are special in the sense that they manipulate expressions and are never present in the expression which results.  For example, note the result of the following command


.. code-block:: python

  > diff(a*b, b)
  a

User functions
~~~~~~~~~~~~~~

.. _ch1__userfunccommands:

.. table:: Commands for user functions.

  +----------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------+
  | Command                                            |   Description                                                                                                             |
  +====================================================+===========================================================================================================================+
  | ``clear(name)``                                    | Clears the name of a user function                                                                                        |
  +----------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------+
  | ``declare(name(arg1, arg2, ...))``                 | declare function name taking dummy arguments ``arg1``, ``arg2``, . . . . Derivatives assumed to be ``0``                  |
  +----------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------+
  | ``define(name(arg1, arg2, ...), obj1, obj2, ...)`` | declare function name taking arguments ``arg1``, ``arg2``, . . . having corresponding derivatives ``obj1``, ``obj2``, ... |
  +----------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------+

Commands for specifying and manipulating user functions are listed in :numref:`ch1__userfunccommands`.  They are used in order to define new user function, as well as the derivatives of the functions with respect to the user variables.  For example, the following expression defines a function named ``f`` which takes one argument.

.. code-block:: python

  > define(f(x), 0.5*x)

The list after the function protoype is used to define the derivatives with respect to each of the independent variables.  Once defined, the function may be used in any other expression.  In addition, any expression can be used as an argument.  For example:

.. code-block:: python

  >>> symdiff('diff(f(x*y),x)')
  '(5.000000000000000e-01 * x * y * y)'
  >>> symdiff('simplify(5.000000000000000e-01 * x * y * y)')
  '(5.000000000000000e-01 * x * pow(y,2)'

.. math::

  \frac{\partial}{\partial x} f \left( u, v, \ldots \right)
    = \frac{\partial u}{\partial x} \cdot \frac{\partial}{\partial u} f \left( u, v, \ldots \right)
    + \frac{\partial v}{\partial x} \cdot \frac{\partial}{\partial v} f \left( u, v, \ldots \right)
    + \ldots

The ``declare`` command is required when the derivatives of two user functions are based on one another.  For example:

.. code::

  >>> symdiff('declare(cos(x))')
  'cos(x)'
  >>> symdiff('define(sin(x),cos(x))')
  'sin(x)'
  >>> symdiff('define(cos(x),-sin(x))')
  'cos(x)'

When declared, a functions derivatives are set to 0, unless specified with a define command.  It is now possible to use these expressions as desired.

.. code::

  >>> symdiff('diff(sin(cos(x)),x)')
  '(-cos(cos(x)) * sin(x))'

Models
~~~~~~

Models are a feature unique to |symdiff|.  They are used to specify symbolic names which have a definition in a symbolic expression.  The most useful property of a model, is that taking the derivative of the model with respect to a parameter results in a new model being created.

For example:

.. code::

  >>> symdiff('define_model(c, (a^2 + b^2)^0.5)')
  'c'
  >>> symdiff('diff(c,a)')
  'c__a'
  >>> symdiff('diff(c,b)')
  'c__b'
  >>> symdiff('model_value(c__a)')
  '(a * pow((pow(a,2) + pow(b,2)),(-5.000000000000000e-01)))'
  >>> symdiff('model_value(c__b)')
  '(b * pow((pow(a,2) + pow(b,2)),(-5.000000000000000e-01)))'


For models which are declared, but not defined, it resolves to a special value of ``UNDEFINED``.  This example is called ``undefined1.py`` in the examples

.. code::

  symdiff('declare_model(y)')
  symdiff('define_model(x, 3 * y + z)')
  symdiff('diff(x, z)')
  print('%s' % symdiff('model_value(x)'))
  print('%s' % symdiff('model_value(x__z)'))
  print('%s' % symdiff('model_value(y__z)'))

.. code::

  ((3 * y) + z)
  (1 + (3 * y__z))
  UNDEFINED

Clearing a model removes its name from the list of models.  Subsequent evaluation of new expressions will treat this name as a variable or function name.  Care should be taken when other models depending on the cleared model remain.

.. code::

  >>> symdiff('declare_model(y)')
  'y'
  >>> symdiff('diff(y,x)')
  'y__x'
  >>> symdiff('clear_model(y)')
  '0'
  >>> symdiff('diff(y,x)')
  '0'

.. _ch1__modelcommands:

.. table:: Commands for models.

  +----------------------------------------------------+---------------------------------+
  | Command                                            |   Description                   |
  +====================================================+=================================+
  | ``clear_model(name)``                              | clear model name                |
  +----------------------------------------------------+---------------------------------+
  | ``declare_model(name)``                            | declare model name              |
  +----------------------------------------------------+---------------------------------+
  | ``define_model(name, exp)``                        | define model having expression  |
  +----------------------------------------------------+---------------------------------+
  | ``model_value(name)``                              | retrieve expression for model   |
  +----------------------------------------------------+---------------------------------+

Commands for specifying and manipulating models are listed in :numref:`ch1__modelcommands`.

Macro Assignment
~~~~~~~~~~~~~~~~

The use of macro assignment allows the substitution of expressions into new expressions.  Every time a command is successfully used, the resulting expression is assigned to a special macro definition, ``$_``.

In this example, the result of the each command is substituted into the next.

.. code::

  >>> symdiff('a + b')
  '(a + b)'
  >>> symdiff('$_ - b')
  '(a + b - b)'
  >>> symdiff('simplify($_)')
  'a'

In addition to the default macro definition, it is possible to specify a variable identifier by using the ``\$`` character followed by an alphanumeric string beginning with a letter.  In addition to letters and numbers, a ``_`` character may be used as well.  A macro which has not previously assigned will implicitly use ``0`` as its value.

This example demonstrates the use of macro assignment.

.. code::

  >>> symdiff('$a1 = a + b')
  '(a + b)'
  >>> symdiff('$a2 = a - b')
  '(a - b)'
  >>> symdiff('simplify($a1+$a2)')
  '(2 * a)'


Handling Exceptions
~~~~~~~~~~~~~~~~~~~

If a |symdiff| evaluation results in an error an exception of type ``symdiff.SymdiffError`` will be thrown.  It may be caught and printed as a string:

.. code::

  try:
    out = symdiff(arg)
  except SymdiffError as x:
    out = x
  print out

Table Output
~~~~~~~~~~~~

``symdiff_table`` command is like the ``symdiff`` command, except that it creates a table for the expression.  The last row is the full expression, and it is made up of entries in the previous rows.  Each column of a row is

0. The name of the operator
1. The operator type
2. A list of indexes for the operator arguments into the table
3. A list of indexes of operators in other rows using this row as an argument 
4. The full string value of the expression composed of this row and the rows it depends on

Example output for the code
 
.. code::
 
  symdiff('declare_model(x)')
  for i, v in enumerate(symdiff_table('a*x + b*c')):
    print '%s %s' % (i, v)
 
is

.. code::

  0 ('a', 'variable', (), (2L,), 'a')
  1 ('x', 'model', (), (2L,), 'x')
  2 ('*', 'product', (0L, 1L), (6L,), '(a * x)')
  3 ('b', 'variable', (), (5L,), 'b')
  4 ('c', 'variable', (), (5L,), 'c')
  5 ('*', 'product', (3L, 4L), (6L,), '(b * c)')
  6 ('+', 'add', (2L, 5L), (), '((a * x) + (b * c))')

