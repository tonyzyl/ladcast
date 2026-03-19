
.. include:: macros.txt

Introduction
------------

Getting Started
~~~~~~~~~~~~~~~
|symdiff|  is a computer algebra tool capable of taking symbolic derivatives.  Using a natural syntax, it is possible to manipulate symbolic equations to aid derivation of equations for a variety of applications.  Additional commands provide the means to simplify results, create common subexpressions, and order expressions for use as source code in a computer program.  With its |python| and |tcl| interpreters, you have the ability to create algorithms to generate equations programatically.

|symdiff| is available from |symdiffurl|.  The source code is available under the terms of the Apache License Version 2.0 :cite:`apache2`. Examples are released under the Apache License Version 2.0 :cite:`apache2`.  Contributions to this project are welcome in the form of bug reporting, documentation, modeling, and feature implementation.

Using the Tool
~~~~~~~~~~~~~~

Interactive Mode
^^^^^^^^^^^^^^^^

The tool is invoked by typing ``symdiff`` on the command line.  On some platforms, the application may also be started by clicking on the application name in a file manager.
A |python| shell is started.  In this introduction, we start by importing the module into the global namespace:

.. code::

  >>> from symdiff import *

We can then start executing |symdiff| commands.

.. code::

   >>> symdiff ('x^y')
   'pow(x,y)'

In this expression, both ``x`` and ``y`` are independent variables.  To differentiate the expression with respect to ``x``, we type the following:

.. code::

   >>> symdiff('diff(x^y,x)')
   '(y * pow(x,(y - 1)))'

By default, any non-numeric string which is not already a function name is treated as an independent variable.

If we wish to simplify the expression, we do the following

.. code::

   >>> symdiff('simplify(x*x+ 2*x^2)')
   '(3 * pow(x,2))'

and to expand out an expression

.. code::

   >>> symdiff('expand((x+ y)*x)')
   '((x * y) + pow(x,2))'

A semicolon ``;``  is used to send multiple commands to the interpreter, but it is not recommended, as it makes debugging more difficult.

Script Mode
^^^^^^^^^^^

With its built-in |python|  interpreter, |symdiff| will execute a script and can be invoked on the command line of your terminal program as:

.. code::

  symdiff myfile.py

where ``myfile.py`` is the name of your input file.

.. _sec__unicode:

Unicode
^^^^^^^

The |python| interpreter, by default, only allows ASCII characters.  In order to enable |unicode| in your |python| scripts, the following line on the first or second line of the script should contain:

.. code::

   # -*- coding: utf-8 -*-

This assumes that the source file is written using \utfeight.  In interactive mode, using |unicode| is not recommended, based on issues in setting the environment properly for the |python| interpreter.

Tcl version
~~~~~~~~~~~

A |tcl| interface to |symdiff| is also available by starting ``symdiff_tcl``.  In order to use |symdiff| in the |tcl| intepreter, the following line is required:

.. code::

   % package require symdifftcl

