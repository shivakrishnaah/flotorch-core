<script setup lang="ts">
const props = withDefaults(defineProps<{
  results?: Record<string, any>[];
  buttonLabel: string;
}>(), {
  results: () => [],
  buttonLabel: 'Download Results',
});

const downloadResults = () => {
  const csv = jsonToCsv(props.results);
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'results.csv';
  a.click();
};
</script>

<template>
  <UButton :label="buttonLabel" icon="i-lucide-download" @click="downloadResults" />
</template>
