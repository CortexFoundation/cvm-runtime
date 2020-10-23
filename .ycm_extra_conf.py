import os
import json
import ycm_core

BUILD_DIRECTORY = 'build'
SRC_LANG = {
    'cuda': ['.cuh', '.cu'],
    'c++': ['.c', '.cc', '.cxx', '.cpp', '.h', '.hpp', '.hxx', 'hh']
}

def DirectoryOfThisScript():
  return os.path.dirname(os.path.abspath(__file__))

def FindCMakeCompilationFile():
    current_dir = DirectoryOfThisScript()
    walk_dirs = [
        current_dir,
        os.path.join(current_dir, BUILD_DIRECTORY),
    ]
    for x in os.listdir(os.path.join(current_dir, BUILD_DIRECTORY)):
        x = os.path.join(current_dir, BUILD_DIRECTORY, x)
        if os.path.isdir(x):
            walk_dirs.append(x)

    db_fname = 'compile_commands.json'
    walk_files = [os.path.join(x, db_fname) for x in walk_dirs]
    files = [x for x in walk_files if os.path.exists(x)]
    return files

def GCC_BIN(flags, binary):
    if 'c++' not in binary:
        return []

    with os.popen(binary + " -dumpversion") as f:
        version = f.readline().strip()

    flag = "-I/usr/include/c++/" + version
    if flag not in flags:
        flags.append(flag)
    return flags

def CXXFLAGS(flags, options):
    CXX_KEYS = ['-I', '-W', '-D', '-m', '-s', '-f']
    for opt in options:
        if opt[:2] in CXX_KEYS and opt not in flags:
            flags.append(opt)
    return flags

def CMakeFlags(flags):
    files = FindCMakeCompilationFile()
    if not files:
        return flags

    with open(files[0], "r") as fin:
        commands = json.load(fin)

    CMAKE_COM_KEY = "command"
    for com in commands:
        if CMAKE_COM_KEY in com:
            com = com[CMAKE_COM_KEY]
            com = [x for x in com.split(' ') if x]
            flags = GCC_BIN(flags, com[0])
            flags = CXXFLAGS(flags, com)
    return flags

def SourceLangFlags(flags, filename):
    ext = os.path.splitext(filename)[-1]
    for lang, suffixs in SRC_LANG.items():
        if ext in suffixs and lang not in flags:
            flags.extend(['-x', lang])
    return flags


COMMON_FLAGS = [ '-I/usr/lib/', '-I/usr/include/']
INIT_FLAGS = {
    'init_ok': False,
    'flags': COMMON_FLAGS,
}
def Settings(**kwargs):
    filename = kwargs['filename']
    if kwargs['language'] != 'cfamily':
        return {}

# def FlagsForFile(filename):
    if not INIT_FLAGS['init_ok']:
        INIT_FLAGS['init_ok'] = True
        INIT_FLAGS['flags'] = CMakeFlags(INIT_FLAGS['flags'])

    final_flags = INIT_FLAGS['flags']
    final_flags = SourceLangFlags(final_flags, filename)

    return {
      'flags': final_flags,
    }
