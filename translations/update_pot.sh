#!/usr/bin/env sh

translations_dir=$(dirname -- "$0")
if [ "$(pwd)" != "$translations_dir" ] ; then
  cd "$translations_dir"
  # go to project root so the file paths in the .pot file will show up nicely
  cd ..
fi
export TZ=UTC

echo "Updating template .pot file"
xgettext \
    --package-name=lada \
    --msgid-bugs-address=https://github.com/ladaapp/issues \
    --from-code=UTF-8 \
    --no-wrap \
    -f <( (find lada/gui -name "*.ui" -or -name "*.py" ; echo lada/cli/main.py) ) \
    -o $translations_dir/lada.pot