<script setup lang="ts">
const project = inject<Ref<Project>>("project");
const projectId = computed(() => project!.value?.id);

// Navigating to project page
navigateTo({
  name: "projects-id",
  params: {
    id: projectId.value
  },
});
</script>

<template>
  <UCard>
    <div class="flex justify-center items-center h-24">
      Navigating to Project {{ projectId }}
    </div>
  </UCard>
</template>