""" BBCode CMD Module Design

    The target is to auto inject command line parser into registry
    table with the python decorator syntax, and the module could
    collect the neccessary options and print usage with `-h` option.

    User try to invoke the main function: `Run` to start customized
    application, which will auto run the appropriate module function.

    >>> from bbcode.common import cmd
    >>> cmd.Run()

    Interfaces
    ==========
    @func Run: Mainly program entry function
    @func module: Module main entry function, the releated function
        will trigger after the command line set name.

        One notable thing to be indicated is that the dependencies
        between modules should not contains cycle, or will raise
        runtime error. The dependent module's options will be treated
        as group reference, and group options will be copied into
        current module's group options.

        @param refs: List of module string, auto combine module options
            by referece to with current module.
    @func group: Group module wrapper function.
    @func option: Wrapper function by the `add_argument` in
        `argparse.ArgumentParser`.

    Notices: The main entry must be zero or one instance, or will raise
        error.

    GroupEntry
    ==========
    A group entry is represented as optional arguments in command line,
    but could execute multi-group main function in single command,
    different from the module entry.

    ModuleEntry
    ===========
    A module entry acts as an sub command at shell, like `git status`.
    The module interface contains `mod_ref` and `mod_main`, which refer
    to the same `CmdEntry` instance. The entry will be achieved via the
    sub parsers method defined in argparse library.

    EntryType
    =========
    The options registered by register_option may be collected as
        cluster, which has permission access, like public, private, ...
        etc. We'd like to support more feasible and extensible usage for
        developers, provided the problems occured in naive coding and
        reported to us.
    @PUBLIC:
    @PRIVATE:

    >>> @option("-p", "--pool-size")
    >>> @module("cmd.test", permission="PRIVATE")
    >>> def cmd_test(args):
    >>>     pass

"""

from __future__ import annotations

from typing import Sequence, Dict, Set, List
from typing import Callable

import copy
import json
import logging
from enum import Enum

import argparse

from .dfs import dfs_visit

__all__ = [
    "PUBLIC", "PRIVATE",
    "module", "group", "option", "global_options",
    "Run"]

class CmdName:
    """ Formatted Cmder Name

        1. Dot splitted name, aka "cmd.test"
        2. Array for module names, aka ["cmd", "test"]
        3. Group related name, aka "cmd-test", always with prefix:"--"
        4. Argument name parsed from command line, aka "cmd_test"
    """

    def __init__(self, log_name : str):
        self.name = str(log_name)

        if isinstance(log_name, CmdName):
            self.name = copy.copy(log_name.name)
        elif isinstance(log_name, list):
            self.name = " ".join(log_name)
        elif isinstance(log_name, str):
            all_spliter = ["-", "_", "."]
            for f in all_spliter:
                if f in log_name:
                    self.name = log_name.replace(f, " ")
                    break

    def __repr__(self):
        return self.name

    # hashable type, can be used at dict type
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other : CmdName):
        if isinstance(other, str):
            other = CmdName(other)
        return other.name == self.name

    @property
    def mod_name(self):
        return self.name.replace(" ", ".")

    @property
    def mod_array(self):
        return [n for n in self.name.split(" ") if n]

    @property
    def mod_prefix_arr(self):
        return [
            ".".join(self.mod_array[:i+1]) \
            for i in range(len(self.mod_array))]

    @property
    def opt_name(self): # command line option name, `--` prefix
        return "--" + self.name.replace(" ", "-")

    @property
    def code_name(self): # code name after argparser parse.
        return self.name.replace(" ", "_")

    @staticmethod
    def topo_sort(cmd_names : Sequence[CmdName]) -> Sequence[CmdName]:
        return sorted(cmd_names, key=lambda x : len(x.name), reverse=True)

PUBLIC = 0
PRIVATE = 1

class CmdOption:
    def __init__(self, *args, **kw):
        self.args = list(args)
        self.kw = dict(kw)

    def __repr__(self):
        return "%s,%s" % (self.args, self.kw)

class GroupOption:
    def __init__(self, permission, entry):
        self.options : Sequence[CmdOption] = []
        self.permission = permission
        self.entry = entry

    def add_option(self, *args, **kw):
        self.options.append(CmdOption(*args, **kw))

    def to_string(self):
        ser = ", ".join([str(o) for o in self.options])
        type_str = "PUBLIC" if self.permission == PUBLIC else "PRIVATE"
        return "%s [%s]" % (type_str, ser)

class CmdFunction:
    def __init__(self, group_opt : GroupOption):
        self.options : GroupOption  = group_opt
        self.func = None

    def __repr__(self):
        func_type = "MAIN" if self.func else "PASS"
        return "%s (%s)" % (func_type, self.func)

    def __call__(self, *args, **kw):
        if self.func is None:
            raise RuntimeError("module {}: null module function".format(
                self.options.entry.name))
        return self.func(*args, **kw)

    def wrapper(self, func):
        if self.func is not None:
            raise RuntimeError("module {}: duplicated functions".format(
                self.options.entry.name))
        self.func = func
        return self

    def move(self) -> CmdFunction:
        """ This function move the wrapper function and relative
                options into other. this will be empty attr after
                invoking.
        """
        move_func = CmdFunction(self.options)
        move_func.func = self.func
        self.func = None
        return move_func

    def empty(self) -> bool:
        return self.func is None

class GroupEntry:
    def __init__(self, name : CmdName):
        self.name = name
        self.options : Sequence[GroupOption] = []
        self.params : CmdOption = CmdOption()
        self.func : CmdFunction = CmdFunction(
            self.group_option(PUBLIC))

    def register_parser(self, *args, **kw):
        self.params.args.extend(args)
        self.params.kw.update(kw)
        return self

    def to_string(self, new_line = False):
        split_str = "\n\t" if new_line else " "
        ser = "name=%s " % self.name.name
        ser += "params=%s " % str(self.params)

        opt_str = ("," + split_str).join([o.to_string() \
            for o in self.options if o.options])
        ser += "options=[%s%s] " % (split_str, opt_str)
        return ser

    def by_main_func(self, permission) -> CmdFunction:
        self.func.options.permission = permission
        return self.func

    def by_pass_func(self, permission) -> CmdFunction:
        return CmdFunction(self.group_option(permission))

    def group_option(self, permission) -> GroupOption:
        self.options.append(GroupOption(permission, self))
        return self.options[-1]

    def public_group_entry(self) -> GroupEntry:
        gentry = GroupEntry(self.name)
        gentry.options = [opt for opt in self.options \
            if opt.permission == PUBLIC]
        gentry.params = self.params
        gentry.func = self.func
        return gentry

class ModEntry(GroupEntry):
    def __init__(self, name : CmdName):
        super(ModEntry, self).__init__(name)
        self.references : Set[str] = set()
        self.groups : Dict[str, GroupEntry] = {}
        # enable global options flag, set via module function
        self.enable_global_opt = True

    def __str__(self):
        return self.name.name

    def to_string(self, new_line = True):
        split_str = "\n\t" if new_line else " "
        ser = super(ModEntry, self).to_string()
        gser = "".join(["," + split_str + v.to_string() \
            for v in self.groups.values()])
        ser += "groups=[%s]" % gser
        return ser

    def group_entry(self, group_name : CmdName) -> GroupEntry:
        assert isinstance(group_name, CmdName)
        if group_name not in self.groups:
            self.groups[group_name] = GroupEntry(group_name)
        return self.groups[group_name]

    def public_group_entry(self) -> GroupEntry:
        gentry = super(ModEntry, self).public_group_entry()

        # disable default module enable option.
        # if not self.func.empty():
        #     gopt = GroupOption(self.func.options.permission, gentry)
        #     gopt.add_option(
        #         self.name.opt_name,
        #         action="store_true",
        #         help="enable module " + str(self.name))
        #     gentry.options.insert(0, gopt)

        gentry.params = copy.deepcopy(self.params)
        gentry.params.args.insert(0, str(self.name))
        # group has no help option
        gentry.params.kw.pop("help", None)
        return gentry

    def as_groups(self):
        common_groups = { self.name: self.public_group_entry() }
        for k, v in self.groups.items():
            common_groups[k] = v.public_group_entry()
        return common_groups

    def update_groups(self, common_groups):
        for k, v in common_groups.items():
            if k not in self.groups:
                self.groups[k] = v

class CmdStorage:
    STORE : Dict[CmdName, ModEntry] = {}
    PARSERS = {}
    GLOBAL_NAME = CmdName("common options")

    @staticmethod
    def get_entry(mod_name, default = None) -> ModEntry:
        if isinstance(mod_name, str):
            mod_name = CmdName(mod_name)

        if mod_name not in CmdStorage.STORE:
            if default is None:
                default = ModEntry(mod_name)
            CmdStorage.STORE[mod_name] = default
        return CmdStorage.STORE[mod_name]

    @staticmethod
    def refs_analysis():
        graph = CmdStorage.STORE.values()
        def refs_generator(entry : ModEntry):
            return [CmdStorage.get_entry(n) for n in entry.references]

        def cycling_trigger(dfs_path : List[ModEntry]):
            dfs_path.append(dfs_path[0])

            common_groups = {}
            for entry in dfs_path[:-1]:
                # TODO: may cause undeterministic behavior, since
                #   group names may be duplicated.
                common_groups.update(entry.as_groups())

            for idx, entry in enumerate(dfs_path[:-1]):
                # remove dependency reference in ref_path
                entry.references.remove(dfs_path[idx+1].name.mod_name)
                entry.update_groups(common_groups)

            dfs_path.pop(-1)

        def visit_func(entry : ModEntry, ref_size : int, index : int):
            if ref_size != index:
                return

            # remove refs and update current entry's groups
            for ref_entry in refs_generator(entry):
                entry.update_groups(ref_entry.as_groups())

            if entry.name in entry.groups:
                del entry.groups[entry.name]

        dfs_visit(
            graph,
            refs_generator,
            cycling_trigger = cycling_trigger,
            visit_func = visit_func)

    @staticmethod
    def init_parser(parser : argparse.ArgumentParser,
                    entry : ModEntry,
                    pre_func):
        logger = logging.getLogger("cmd.parser")

        has_main_entry = not entry.func.empty()
        for group in entry.groups.values():
            if not group.func.empty():
                has_main_entry = True
        # There is no need to create group options and options
        if not has_main_entry:
            return

        for group in entry.groups.values():
            # skip empty group options
            if not group.options:
                continue

            try:
                gparser = parser.add_argument_group(
                    *group.params.args, **group.params.kw)
            except Exception as e:
                logger.error("module({}):group({}): {}".format(
                    entry.name, group.name, e))
                raise e
            for gopt in group.options:
                for opt in gopt.options:
                    try:
                        gparser.add_argument(*opt.args, **opt.kw)
                    except argparse.ArgumentError as e:
                        logger.error("module({}):group({}): {}".format(
                            entry.name, group.name, e))
                        raise e

        for gopt in entry.options:
            for opt in gopt.options:
                try:
                    parser.add_argument(*opt.args, **opt.kw)
                except argparse.ArgumentError as e:
                    logger.error("module({}): {}".format(
                        entry.name, e))
                    raise e

        def _func(args):
            # invoke prepare function
            if entry.enable_global_opt and not pre_func.empty():
                pre_func(args)

            for group in entry.groups.values():
                if group.func.empty():
                    continue
                if getattr(args, group.name.code_name, None):
                    return group.func(args)

            if not entry.func.empty():
                return entry.func(args)

            raise RuntimeError(
                "can not find module [" + entry.name.name + \
                "] main function to run")
        parser.set_defaults(func=_func)

    @staticmethod
    def init_parser_object(parser_object, mod_name, entry, pre_func):
        logger = logging.getLogger("cmd.parser")

        # add subparser
        if "sub_parser" not in parser_object:
            parser_object["sub_parser"] = \
                parser_object["parser"].add_subparsers(
                    title = "COMMAND",
                    description = "supportive sub commands")

        entry.params.kw.setdefault(
            "formatter_class",
            argparse.RawDescriptionHelpFormatter)

        try:
            mod_parser = parser_object["sub_parser"].add_parser(
                mod_name, *entry.params.args, **entry.params.kw)
        except Exception as e:
            logger.error("module({}): {}".format(
                entry.name, e))
            raise e

        CmdStorage.init_parser(mod_parser, entry, pre_func)
        parser_object[mod_name] = { "parser": mod_parser, }
        return mod_parser

    @staticmethod
    def init_parsers() -> argparse.ArgumentParser:
        CmdStorage.refs_analysis()

        pre_entry = CmdStorage.get_entry(CmdStorage.GLOBAL_NAME)
        CmdStorage.STORE.pop(CmdStorage.GLOBAL_NAME)
        pre_func = pre_entry.func.move()
        pre_groups = pre_entry.as_groups()

        # remove unuseful module path
        for name in CmdName.topo_sort(CmdStorage.STORE.keys()):
            entry = CmdStorage.STORE[name]
            if getattr(entry, "has_main_entry", None):
                continue
            has_main_entry = not entry.func.empty()
            for group in entry.groups.values():
                if not group.func.empty():
                    has_main_entry = True

            if has_main_entry:
                for mod_name in entry.name.mod_prefix_arr:
                    if mod_name in CmdStorage.STORE:
                        setattr(CmdStorage.get_entry(mod_name),
                                "has_main_entry", True)
            else:
                del CmdStorage.STORE[name]

        # init root parser descriptions
        root_entry = CmdStorage.get_entry("")
        root_entry.params.kw.setdefault(
            "description",
            "bbcode helper script, implemented via python3")
        root_entry.params.kw.setdefault(
            "formatter_class",
            argparse.RawDescriptionHelpFormatter)
        if root_entry.enable_global_opt:
            root_entry.update_groups(pre_groups)

        logger = logging.getLogger("cmd.parser")
        try:
            root_parser = argparse.ArgumentParser(
                *root_entry.params.args, **root_entry.params.kw)
        except Exception as e:
            logger.error("module({}): {}".format(
                root_entry.name, e))
            raise e

        CmdStorage.init_parser(root_parser, root_entry, pre_func)

        # set root parser object
        CmdStorage.PARSERS["parser"] = root_parser

        for entry in list(CmdStorage.STORE.values()):
            parser_object = CmdStorage.PARSERS
            for mod_name, prefix in zip(
                    entry.name.mod_array, entry.name.mod_prefix_arr):
                if mod_name not in parser_object:
                    mod_entry = CmdStorage.get_entry(prefix)
                    if mod_entry.enable_global_opt:
                        mod_entry.update_groups(pre_groups)
                    CmdStorage.init_parser_object(
                        parser_object, mod_name,
                        mod_entry, pre_func)
                parser_object = parser_object[mod_name]
        return root_parser

    @staticmethod
    def get_parser(parser_path) -> argparse.ArgumentParser:
        parser_object = CmdStorage.PARSERS
        for mod_name in CmdName(parser_path).mod_array:
            if mod_name not in parser_object:
                raise RuntimeError("cannot find parser: " + parser_path)
            parser_object = parser_object[mod_name]
        return parser_object["parser"]


""" CMD Registration API
"""

def option(*args, **kw):
    """ ArgParse:add_argument function wrapper options

        Parameters
        ==========

        name or flags: aka "-f", "--foo"
        action: available options are
            "store"(default),
            "store_const",
            "store_true", "store_false",
            "append", "append_const",
            "count",
            "help"(disabled),
            "version", "extend"
        nargs:
        const:
        default: store default value, None by default
        type: argument type
        choices: available options
        required: make optional argument required like "-f"
        help: print help information
        metavar: meta variable in usage
        dest:

    """
    def _func(func : CmdFunction):
        func.options.add_option(*args, **kw)
        return func
    return _func

def module(mod_name, *args,
        as_main = False, refs = [],
        with_global_opt = True,
        permission = PUBLIC, **kw):
    """ Module Interface

        Root Parser Params
        ==================
        prog - The name of the program (default: sys.argv[0])
        usage - The string describing the program usage
            (default: generated from arguments added to parser)
        description - Text to display before the argument help
            (default: none)
        epilog - Text to display after the argument help (default: none)
        parents - A list of ArgumentParser objects whose arguments
            should also be included
        formatter_class - A class for customizing the help output
        prefix_chars - The set of characters that prefix optional
            arguments (default: ‘-‘)
        fromfile_prefix_chars - The set of characters that prefix
            files from which additional arguments should be read
            (default: None)
        argument_default - The global default value for arguments
            (default: None)
        conflict_handler - The strategy for resolving conflicting
            optionals (usually unnecessary)
        add_help - Add a -h/--help option to the parser (default: True)
        allow_abbrev - Allows long options to be abbreviated if
            the abbreviation is unambiguous. (default: True)
        exit_on_error - Determines whether or not ArgumentParser
            exits with error info when an error occurs. (default: True)

        Sub Module Params
        =================
        title - title for the sub-parser group in help output;
            by default “subcommands” if description is provided,
            otherwise uses title for positional arguments
        description - description for the sub-parser group in help
            output, by default None
        prog - usage information that will be displayed with
            sub-command help, by default the name of the program
            and any positional arguments before the subparser argument
        parser_class - class which will be used to create sub-parser
            instances, by default the class of the current parser
            (e.g. ArgumentParser)
        action - the basic type of action to be taken when this
            argument is encountered at the command line
        dest - name of the attribute under which sub-command name
            will be stored; by default None and no value is stored
        required - Whether or not a subcommand must be provided,
            by default False (added in 3.7)
        help - help for sub-parser group in help output,
            by default None
        metavar - string presenting available sub-commands in help;
            by default it is None and presents sub-commands in
            form {cmd1, cmd2, ..}
    """

    mod_entry = CmdStorage.get_entry(mod_name)
    mod_entry.enable_global_opt = with_global_opt
    mod_entry.references.update(refs)
    mod_entry.register_parser(*args, **kw)
    if as_main:
        return mod_entry.by_main_func(permission).wrapper
    return mod_entry.by_pass_func(permission).wrapper

def group(mod_name,
          as_main = False, refs = [],
          permission = PRIVATE,
          # group parameters
          group_name = None, with_short=False,
          description=None):
    mod_entry = CmdStorage.get_entry(mod_name)
    mod_entry.references.update(refs)
    def _func(func):
        gname = CmdName(func.__name__)
        if group_name is not None:
            gname = CmdName(group_name)
        gentry = mod_entry.group_entry(gname)

        desc = description if description else func.__doc__
        gentry.register_parser(
            title=str(gentry.name),
            description=desc)

        if as_main:
            gfunc = gentry.by_main_func(permission)
            short_opt = "-{}".format(gname.name[0])
            opt_args = [short_opt] if with_short else []
            opt_args.append(gentry.name.opt_name)
            gfunc.options.add_option(
                *opt_args,
                action="store_true",
                help="enable module " + str(gentry.name))
        else:
            gfunc = gentry.by_pass_func(permission)
        return gfunc.wrapper(func)
    return _func

def global_options(refs=[]):
    return module(CmdStorage.GLOBAL_NAME, refs=refs, as_main=True)

def parser(name) -> argparse.ArgumentParser:
    parser_object = CmdStorage.PARSERS
    for mod_name in CmdName(name).mod_array:
        if mod_name in parser_object:
            parser_object = parser_object[mod_name]
        else:
            raise RuntimeError("parser:{} not found".format(name))
    return parser_object["parser"]

# Convenient class for user to create args
class Args:
    pass

def Run():
    root_parser = CmdStorage.init_parsers()
    args = root_parser.parse_args()

    if getattr(args, "func", None):
        args.func(args)
        return args

    raise RuntimeError(
        "cannot find the mainly function to run, " +
        "please set main function via mod_main or group_main."
    )
