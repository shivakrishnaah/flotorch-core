<script setup lang="ts">

const props = defineProps<{
  fieldName: keyof typeof useTooltipInfo
}>()

const setTooltipInfo = computed(() => {
  return useTooltipInfo[props.fieldName];
})

</script>



<template>
  <UTooltip :text="setTooltipInfo">
    <UButton icon="i-lucide-info" color="neutral" variant="ghost" />
  </UTooltip>
</template>
