#!/usr/bin/env ruby

require "optparse"

options = { apply: false, root: ".", map: "CONFIG_RENAME_MAP.tsv" }
OptionParser.new do |parser|
  parser.on("--apply") { options[:apply] = true }
  parser.on("--root DIR") { |dir| options[:root] = dir }
  parser.on("--map FILE") { |file| options[:map] = file }
end.parse!

root = File.expand_path(options[:root])
map_path = File.expand_path(options[:map], root)
abort("Rename map does not exist: #{map_path}") unless File.file?(map_path)

mappings = File.readlines(map_path, chomp: true).drop(1).map do |line|
  old_name, new_name = line.split("\t", 2)
  abort("Invalid mapping line: #{line}") unless old_name && new_name
  [old_name, new_name]
end

extensions = %w[.sh .md .py .rb .tex .yaml .yml .txt]
excluded = [
  File.expand_path("CONFIG_RENAME_MAP.tsv", root),
  File.expand_path("EXPERIMENT_REGISTRY.md", root),
]
changed = []

Dir.glob(File.join(root, "**", "*"), File::FNM_DOTMATCH).sort.each do |path|
  next unless File.file?(path)
  next if path.include?("/.git/")
  next if excluded.include?(path)
  next unless extensions.include?(File.extname(path))

  original = File.binread(path)
  updated = original.dup
  mappings.each do |old_name, new_name|
    old_base = File.basename(old_name, ".yaml")
    new_base = File.basename(new_name, ".yaml")
    updated = updated.gsub("configs/#{old_name}", "configs/#{new_name}")
    updated = updated.gsub("--exp-name #{old_base}", "--exp-name #{new_base}")
    updated = updated.gsub(%Q{EXP_NAME="#{old_base}"}, %Q{EXP_NAME="#{new_base}"})
  end
  next if updated == original

  changed << path.sub("#{root}/", "")
  File.binwrite(path, updated) if options[:apply]
end

verb = options[:apply] ? "UPDATED" : "WOULD UPDATE"
changed.each { |path| puts "#{verb} #{path}" }
puts "#{verb} #{changed.length} files"
