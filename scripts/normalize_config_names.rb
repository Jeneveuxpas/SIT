#!/usr/bin/env ruby

require "fileutils"
require "optparse"
require "yaml"

options = { apply: false, config_dir: "configs" }
OptionParser.new do |parser|
  parser.on("--apply") { options[:apply] = true }
  parser.on("--config-dir DIR") { |dir| options[:config_dir] = dir }
end.parse!

config_dir = File.expand_path(options[:config_dir])
abort("Config directory does not exist: #{config_dir}") unless Dir.exist?(config_dir)

DEFAULTS = {
  "model" => "SiT-B/2-EncoderKV",
  "encoder-depth" => 8,
  "projection-layer-type" => "conv",
  "enc-layer-indices" => "11",
  "sit-layer-indices" => "10",
  "stage1-steps" => 30_000,
  "transition-steps" => 0,
  "transition-schedule" => "cosine",
  "transition-blend-mode" => "output",
  "distill-coeff" => 1.0,
  "distill-warmup-steps" => 0,
  "distill-warmup-schedule" => "linear",
  "align-mode" => "attn_mse",
  "kv-proj-type" => "linear",
  "kv-norm-type" => "none",
  "kv-replace-mode" => "kv",
  "kv-memory-mode" => "replace",
  "kv-stop-step" => -1,
  "kv-stop-fade-steps" => 0,
  "scaffold-interface" => "kv",
  "scaffold-feature-source" => "attn_input",
  "proj-coeff" => "1.0",
  "repa-loss" => true,
  "use-kv" => true,
  "projection-loss-type" => "cosine",
  "spnorm-method" => "zscore",
  "zscore-alpha" => 1.0,
  "path-type" => "linear",
  "prediction" => "v",
  "resolution" => 256,
}.freeze

def value(config, key)
  config.fetch(key, DEFAULTS.fetch(key, nil))
end

def number_token(number)
  number = Float(number)
  return number.to_i.to_s if number == number.to_i

  number.to_s.sub("-", "m").tr(".", "p")
end

def step_token(number)
  number = Integer(number)
  return "#{number / 1_000_000}m" if number >= 1_000_000 && (number % 1_000_000).zero?
  return "#{number / 1_000}k" if number >= 1_000 && (number % 1_000).zero?

  number.to_s
end

def model_token(model)
  case model
  when /SiT-XL\/2/ then "sit-xl2"
  when /SiT-L\/2/ then "sit-l2"
  when /SiT-B\/2/ then "sit-b2"
  else model.downcase.gsub(/[^a-z0-9]+/, "-").gsub(/^-|-$/, "")
  end
end

def encoder_token(encoder)
  {
    "dinov2-b" => "dinob",
    "dinov2-vit-b" => "dinob",
    "clip-B" => "clipb",
    "clip-L" => "clipl14",
    "mae-b" => "maeb",
    "mae-l" => "mael",
    "deit3-b" => "deit3b",
    "sam2-s" => "sam2s",
  }.fetch(encoder, encoder.to_s.downcase.gsub(/[^a-z0-9]+/, ""))
end

def loss_token(loss)
  {
    "mse_v_norm" => "msevnorm",
    "cosine_noisy" => "cosinenoisy",
    "cosine_norm" => "cosinenorm",
  }.fetch(loss, loss.to_s.downcase.gsub(/[^a-z0-9]+/, ""))
end

def consistency_token(mode, coefficient)
  target = {
    "attn_mse" => "attn",
    "kv_mse" => "kv",
    "attn_cosine" => "attncos",
  }.fetch(mode, mode.to_s.downcase.gsub(/[^a-z0-9]+/, ""))
  "cons-#{target}#{number_token(coefficient)}"
end

def canonical_name(config)
  model = value(config, "model")
  repa = value(config, "repa-loss") == true
  use_kv = value(config, "use-kv") == true && model.include?("EncoderKV")
  stage_steps = Integer(value(config, "stage1-steps") || 0)
  distill_coeff = Float(value(config, "distill-coeff") || 0.0)
  effective_scaffold = use_kv && (stage_steps.positive? || distill_coeff.positive?)

  parts = [model_token(model)]
  if effective_scaffold
    encoder = encoder_token(value(config, "enc-type"))
    encoder_layers = value(config, "enc-layer-indices").to_s.delete(" ").tr(",", "-")
    parts += ["attnscaf", "#{encoder}#{encoder_layers}"]

    interface = value(config, "scaffold-interface")
    source = value(config, "scaffold-feature-source")
    replace = value(config, "kv-replace-mode")
    memory = value(config, "kv-memory-mode")
    parts << "if-#{interface}" unless interface == "kv"
    parts << "src-#{source.gsub("_", "")}" unless source == "attn_input"
    parts << "replace-#{replace}" unless replace == "kv"
    parts << "memory-#{memory.gsub("_", "")}" unless memory == "replace"
    if config["encoder-patch-shuffle"] == true
      grid = config["encoder-patch-shuffle-grid"]
      patch = config["encoder-patch-shuffle-patch-size"]
      shuffle = "shuffle"
      shuffle += "-g#{grid}" if grid
      shuffle += "-p#{patch}" if patch
      parts << shuffle
    end

    sit_layers = value(config, "sit-layer-indices").to_s.delete(" ").tr(",", "-")
    parts += ["s#{sit_layers}", "t#{step_token(stage_steps)}"]
    parts << (distill_coeff.zero? ? "cons-none" : consistency_token(value(config, "align-mode"), distill_coeff))

    transition_steps = Integer(value(config, "transition-steps") || 0)
    if transition_steps.positive?
      blend = value(config, "transition-blend-mode") == "kv" ? "kv" : "attn"
      schedule = value(config, "transition-schedule")
      smooth = "smooth-#{blend}"
      smooth += "-#{schedule}" unless schedule == "cosine"
      parts << smooth
      parts << "smoothsteps-#{step_token(transition_steps)}" unless transition_steps == 5_000
    else
      parts << "smooth-none"
    end

    warmup = Integer(value(config, "distill-warmup-steps") || 0)
    parts << "cwarm#{step_token(warmup)}" if warmup.positive?
    stop = config["kv-stop-step"]
    if distill_coeff.positive? && stop && Integer(stop) >= 0
      parts << "cstop#{step_token(stop)}"
      fade = Integer(config.fetch("kv-stop-fade-steps", 0))
      parts << "cfade#{step_token(fade)}" if fade.positive?
    end

    kv_proj = value(config, "kv-proj-type")
    kv_norm = value(config, "kv-norm-type")
    parts << "kvproj-#{kv_proj.gsub("_", "")}" unless kv_proj == "linear"
    parts << "kvnorm-#{kv_norm.gsub("_", "")}" unless kv_norm == "none"
  elsif repa
    parts << "repa-only"
    encoder = config["enc-type"]
    parts << encoder_token(encoder) if encoder
  else
    parts << "vanilla"
  end

  if repa
    depth = Integer(value(config, "encoder-depth"))
    coeff = number_token(value(config, "proj-coeff"))
    projection_loss = value(config, "projection-loss-type")
    norm = projection_loss == "cosine_repa" ? "none" : value(config, "spnorm-method")
    norm_token = case norm
                 when "none" then "none"
                 when "zscore" then "zs#{number_token(value(config, "zscore-alpha"))}"
                 when "zscore_token" then "zstoken#{number_token(value(config, "zscore-alpha"))}"
                 else norm.to_s.gsub("_", "")
                 end
    parts += ["repa#{depth}-#{coeff}", "norm-#{norm_token}"]
    parts << "rloss-#{loss_token(projection_loss)}" unless ["cosine", "cosine_repa"].include?(projection_loss)
    projector = config["projection-layer-type"]
    parts << "rproj-#{projector}" if projector

    stop = config["repa-stop-step"]
    if stop && Integer(stop) >= 0
      parts << "rstop#{step_token(stop)}"
      fade = Integer(config.fetch("repa-stop-fade-steps", 0))
      parts << "rfade#{step_token(fade)}" if fade.positive?
    end
  else
    parts << "repa-none"
  end

  parts << "path-cosine" if value(config, "path-type") == "cosine"
  parts << "res#{value(config, "resolution")}" unless Integer(value(config, "resolution")) == 256
  parts << "vae-#{config["vae"]}" if config["vae"]
  parts.join("-") + ".yaml"
end

files = Dir[File.join(config_dir, "*.yaml")].sort
excluded_losses = %w[cosine_v mse_v_norm]
deletions = Hash.new { |hash, key| hash[key] = [] }
records = []

files.each do |path|
  config = YAML.safe_load(File.read(path)) || {}
  projection_loss = config["projection-loss-type"]
  if excluded_losses.include?(projection_loss)
    deletions[projection_loss] << path
    next
  end

  target_name = canonical_name(config)
  target = File.join(config_dir, target_name)
  records << [path, target, config]
end

runtime_keys = %w[epochs max-train-steps checkpointing-steps sampling-steps n-samples]
normalize = lambda do |config|
  DEFAULTS.merge(config).reject { |key, _| runtime_keys.include?(key) }
end

renames = []
duplicate_deletions = []
records.group_by { |_, target, _| target }.each do |target, group|
  if group.length == 1
    source, = group.first
    renames << [source, target] unless source == target
    next
  end

  normalized = group.map { |_, _, config| normalize.call(config) }
  unless normalized.all? { |config| config == normalized.first }
    sources = group.map { |source, _, _| "  #{source}" }.join("\n")
    abort("Canonical-name collision between distinct conditions:\n#{sources}\n  -> #{target}")
  end

  winner = group.max_by do |source, _, config|
    [Integer(config.fetch("max-train-steps", DEFAULTS.fetch("max-train-steps", 100_000))), source]
  end
  winner_source, = winner
  group.each do |source, _, _|
    duplicate_deletions << source unless source == winner_source
  end
  renames << [winner_source, target] unless winner_source == target
  group.each do |source, _, _|
    renames << [source, target] if source != winner_source
  end

  puts "MERGE #{group.length} equivalent configs -> #{File.basename(target)}"
end

excluded_losses.each do |loss|
  puts "DELETE #{deletions[loss].length} #{loss} configs"
  deletions[loss].each { |path| puts "  #{File.basename(path)}" }
end
puts "DELETE #{duplicate_deletions.length} superseded duplicate configs"
duplicate_deletions.each { |path| puts "  #{File.basename(path)}" }
puts "RENAME #{renames.length} configs"
renames.each { |source, target| puts "  #{File.basename(source)} -> #{File.basename(target)}" }

exit unless options[:apply]

deletions.each_value { |paths| paths.each { |path| FileUtils.rm(path) } }
duplicate_deletions.each { |path| FileUtils.rm(path) }

temporary = []
physical_renames = renames.reject { |source, _| duplicate_deletions.include?(source) }
physical_renames.each_with_index do |(source, target), index|
  staged = File.join(config_dir, ".config-rename-#{index}.yaml")
  FileUtils.mv(source, staged)
  temporary << [staged, target]
end
temporary.each { |staged, target| FileUtils.mv(staged, target) }

map_path = File.join(File.dirname(config_dir), "CONFIG_RENAME_MAP.tsv")
existing_mappings = []
if File.exist?(map_path)
  existing_mappings = File.readlines(map_path, chomp: true).drop(1).each_with_object([]) do |line, mappings|
    old_name, new_name = line.split("\t", 2)
    mappings << [old_name, new_name] if old_name && new_name
  end
end
all_mappings = (existing_mappings + renames.map do |source, target|
  [File.basename(source), File.basename(target)]
end).uniq.select { |_, new_name| File.exist?(File.join(config_dir, new_name)) }.sort_by(&:first)
File.open(map_path, "w") do |file|
  file.puts("old_config\tnew_config")
  all_mappings.each do |old_name, new_name|
    file.puts("#{old_name}\t#{new_name}")
  end
end

puts "WROTE #{map_path}"
