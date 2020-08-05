
*********************************
Write Documentation and Tutorials
*********************************

.. content:

.. _write_math_formalization:

Math Formalization
==================

- source: docs/guide/arch.rst

The mathematic formalization doc use three-phase format,
followed by two back-slash in the end-line, which
indicates the generated math equation starting from new line.
Code refer to this:

.. code-block:: Text
  
  Y[<y_indices>] = X[<x_indices>], \\

  \forall <given range>, \\

  \text{where }
    <condition1> \text{ and} (\\)
    <condition2> \text{ and} (\\)
    ...
    <conditionn>

Above syntax wrapped by the angle bracket will be replaced by
writers' specific content. And more detail requisites are
introduced at following sections.

The round brackets are optional if the equation preview is short.

Array and Set
-------------

The concept of Array is a indexed set. Array may have multi
dimension, likes :math:`[n_1, n_2, \cdots, n_N]`, and Set is a
unordered union with no-duplicate values.

Array to Set can be unidirectional, removing the index and
duplicated values. In mathematic formalization, the cast from
Array to Set is implicit and auto sometimes.

We use the captial letter to indicates Array and Set usually.

Map Set
-------

Sometimes it's hard and pointless to descripe the mathematic
logic, and we can define the Set map to give the neccessary
property it comes out. Keep to the format:

.. code-block:: Text

  <map name>: <Set1> \to <Set2>, \text{s.t. }
    <constraints>

.. note::
  
  Suppose the possible variable in constaints have already
  satisfied the domain of definition: Set1. 
  And we usually ignore the conditions by default.

MACRO and NAME
--------------

Define and reference the macro or name with text format:

.. code-block:: Text

  \text{MACRO_NAME}


Indent
------

Indent use two whitespaces, and multiline directives like
`begin{cases}`, `union map` should set the proper recursive
ident for easier read. Code refer to this:

.. code-block:: Text

  R = \begin{cases}
    U - T, & \text{if exclude is true} \\
    T, & \text{otherwise}
  \end{cases}

  I: [0, N-r) \to U-R, \text{s.t. }
    \forall i < j, I(i) < I(j) \text{ and} \\

Mathjax Directives
------------------

- \\begin {cases} \\end {cases}

  .. math::
    x = \begin{cases}
      i, & \text{if } i\geqslant 0 \\
      i + N, & \text{otherwise}
    \end{cases}
    
- \\wedge: :math:`\wedge`
- \\cdots: :math:`\cdots`

