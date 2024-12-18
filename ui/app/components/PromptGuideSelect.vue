<script setup lang="ts">
import type { FormError } from '@nuxt/ui';
import { z } from 'zod'
const modelValue = defineModel<ProjectNShotPromptGuide>()
const props = defineProps<{
  requiredPrompts: number
}>()

const { open, reset, onChange } = useFileDialog({
  accept: "application/json",
  directory: false,
})

const validationSchema = computed(() => {
  const schema = ProjectNShotPromptGuideSchema.extend({
    examples: z.array(z.object({
      example: z.string(),
    })).min(props.requiredPrompts, {
      message: `Must have at least ${props.requiredPrompts} examples`
    })
  })
  if (props.requiredPrompts > 0) {
    return schema
  }
  return schema.optional()
})

const emits = defineEmits<{
  error: [FormError | undefined]
}>()

onChange(async (files) => {
  if (!files || files.length === 0) {
    return
  }
  const file = files[0]
  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const json = JSON.parse(e.target?.result as string)
      const parsed = validationSchema.value.safeParse(json)
      if (parsed.success) {
        modelValue.value = parsed.data
        filepath.value = "Attached"
        emits("error", undefined)
      } else {
        const errorMessage = [parsed.error.flatten().fieldErrors.examples?.[0], parsed.error.flatten().fieldErrors.user_prompt?.[0], parsed.error.flatten().fieldErrors.system_prompt?.[0]].filter(Boolean).join("\n")
        emits("error", { name: "n_shot_prompt_guide", message: errorMessage })
      }
    } catch (error) {
      console.error(error)
      emits("error", { name: "n_shot_prompt_guide", message: "Invalid file format" })
    }
  }
  reader.readAsText(file!, "UTF-8")
  reset()
})

const filepath = ref('')
const isUploading = ref(false)

</script>



<template>
  <UButtonGroup class="w-full">
    <UInput v-model="filepath" disabled />
    <UButton color="neutral" variant="subtle" icon="i-lucide-upload" :loading="isUploading" @click="open()" />
  </UButtonGroup>
</template>
