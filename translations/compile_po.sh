#!/usr/bin/env sh

translations_dir=$(dirname -- "$0")
if [ "$(pwd)" != "$translations_dir" ] ; then
  cd "$translations_dir"
fi

find . -mindepth 1 -maxdepth 1 -type f -name "*.po" -printf '%f\n' | while read po_file ; do
  lang="${po_file%.po}"
  if [ ! -d "$lang/LC_MESSAGES" ] ; then
    mkdir -p "$lang/LC_MESSAGES"
  fi
  echo "Compiling language $lang .po file into .mo file"
  msgfmt "$po_file" -o "$lang/LC_MESSAGES/lada.mo"
done