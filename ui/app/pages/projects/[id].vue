<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query';
const projectId = useRouteParams<string>("id");

const { data: project } = useQuery({
  queryKey: ["projects", projectId],
  queryFn: () => useProject(projectId.value)
})

provide("project", project)

const title = computed(() => project!.value?.name || `Project - ${projectId.value}`)

const description = computed(() => {
  const string = "Project Status: " + useHumanProjectStatus(project!.value?.status, "Loading...")
  return string
})

const route = useRoute()

// const showSetupExperimentsButton = computed(() => project!.value?.status === "not_started" && route.name === "projects-id")

const { isInfraAlertHidden } = useInfraAlertState()

const handleAlertClose = () => {
  isInfraAlertHidden.value = true
}


const hideInfraAlert = computed(() => {
  if (isInfraAlertHidden.value) {
    return true
  }
  if (route.matched.some((route) => route.meta.hideInfraAlert)) {
    return true
  }
  return false
})
</script>



<template>
  <div>
    <div v-if="!hideInfraAlert" class="mb-4">
      <UAlert variant="subtle" title="Heads Up!" :close="{
        onClick: () => {
          handleAlertClose()
        }
      }"
        description="The infrastructure provisioned for FloTorch experiments has to be explicitly deprovisioned to avoid incurring continued AWS billing" />

    </div>
    <Page v-if="project" :title="title" :to="{ name: 'projects-id', params: { id: projectId } }"
      :description="description">
      <!-- <template #actions>
        <UButton v-if="showSetupExperimentsButton" :to="{
          name: 'projects-id-validexperiments',
          params: {
            id: projectId
          }
        }" icon="i-lucide-play" label="Execute Run" />
      </template> -->
      <NuxtPage />
    </Page>
  </div>
</template>
