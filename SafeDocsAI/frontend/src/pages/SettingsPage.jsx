import React, { useCallback, useEffect, useState } from 'react';
import { settingsService } from '../services/settingsService';
import { Button } from '../components/ui/Button';
import { useLocale } from '../i18n';
import { formatLocaleDate } from '../lib/locale';

const ROLE_OPTIONS = ['admin', 'content_manager', 'user'];

const normalizeSelectedModel = (currentValue, availableModels) => {
    if (!Array.isArray(availableModels) || availableModels.length === 0) {
        return '';
    }

    return availableModels.includes(currentValue) ? currentValue : availableModels[0];
};

const DEFAULT_RUNTIME_SETTINGS = {
    chat_model: '',
    embedding_model: '',
    enable_condense_query: true,
    contextual_embedding_enabled: false,
    contextual_embedding_model: '',
    top_k: 10,
    chat_model_num_ctx: 20000,
    contextual_embedding_num_ctx: 8192,
    reranker_enabled: false,
    reranker_model: 'gemma4:e4b',
    available_models: [],
    available_chat_models: [],
    available_embedding_models: [],
    ollama_available: true,
    ollama_error: '',
};

const buildRuntimeSettingsState = (settingsData = {}, previousSettings = {}, overrides = {}) => {
    const next = {
        ...DEFAULT_RUNTIME_SETTINGS,
        ...previousSettings,
        ...settingsData,
        ...overrides,
    };

    return {
        ...next,
        chat_model: normalizeSelectedModel(
            next.chat_model || next.model || '',
            next.available_chat_models || []
        ),
        embedding_model: normalizeSelectedModel(
            next.embedding_model || '',
            next.available_embedding_models || []
        ),
        enable_condense_query: Boolean(next.enable_condense_query),
        contextual_embedding_enabled: Boolean(next.contextual_embedding_enabled),
        reranker_enabled: Boolean(next.reranker_enabled),
        top_k: Number(next.top_k) || DEFAULT_RUNTIME_SETTINGS.top_k,
        chat_model_num_ctx: Number(next.chat_model_num_ctx) || DEFAULT_RUNTIME_SETTINGS.chat_model_num_ctx,
        contextual_embedding_num_ctx: Number(next.contextual_embedding_num_ctx) || DEFAULT_RUNTIME_SETTINGS.contextual_embedding_num_ctx,
        reranker_model: next.reranker_model || DEFAULT_RUNTIME_SETTINGS.reranker_model,
    };
};

const SettingsPage = () => {
    const { locale, t } = useLocale();
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [runtimeSettings, setRuntimeSettings] = useState(DEFAULT_RUNTIME_SETTINGS);
    const [users, setUsers] = useState([]);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const canSave = Boolean(runtimeSettings.chat_model && runtimeSettings.embedding_model);

    const loadData = useCallback(async () => {
        setIsLoading(true);
        setError('');

        try {
            const [settingsRes, usersRes] = await Promise.all([
                settingsService.get(),
                settingsService.getUsers(),
            ]);
            const settingsData = settingsRes.data || {};
            setRuntimeSettings((prev) => buildRuntimeSettingsState(settingsData, prev));
            setUsers(usersRes.data || []);
        } catch (err) {
            console.error('Failed to load settings', err);
            setError(err.response?.data?.detail || t('settings.loadFailed'));
        } finally {
            setIsLoading(false);
        }
    }, [t]);

    useEffect(() => {
        loadData();
    }, [loadData]);

    const handleSave = async () => {
        setIsSaving(true);
        setMessage('');
        setError('');

        try {
            const payload = {
                chat_model: runtimeSettings.chat_model,
                embedding_model: runtimeSettings.embedding_model,
                enable_condense_query: runtimeSettings.enable_condense_query,
                contextual_embedding_enabled: runtimeSettings.contextual_embedding_enabled,
                contextual_embedding_model: runtimeSettings.contextual_embedding_model,
                top_k: Number(runtimeSettings.top_k) || 10,
                chat_model_num_ctx: Number(runtimeSettings.chat_model_num_ctx) || 20000,
                contextual_embedding_num_ctx: Number(runtimeSettings.contextual_embedding_num_ctx) || 8192,
                reranker_enabled: Boolean(runtimeSettings.reranker_enabled),
                reranker_model: runtimeSettings.reranker_model || 'gemma4:e4b',
            };
            const response = await settingsService.update(payload);
            const settingsData = response.data || {};
            setRuntimeSettings((prev) => buildRuntimeSettingsState(settingsData, prev, payload));
            setMessage(t('settings.saved'));
        } catch (err) {
            console.error('Failed to update settings', err);
            setError(err.response?.data?.detail || t('settings.saveFailed'));
        } finally {
            setIsSaving(false);
        }
    };

    const handleRoleChange = async (userId, role) => {
        try {
            await settingsService.updateUserRole(userId, role);
            setUsers((prev) => prev.map((user) => (
                user.id === userId ? { ...user, role } : user
            )));
        } catch (err) {
            console.error('Failed to update role', err);
            alert(err.response?.data?.detail || t('settings.roleUpdateFailed'));
        }
    };

    if (isLoading) {
        return (
                <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center text-slate-500 shadow-sm">
                {t('settings.loading')}
            </div>
        );
    }

    return (
        <div className="space-y-6 px-4">
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="mb-4 flex items-center gap-3">
                    <h2 className="text-3xl font-extrabold text-[#1f3a60]">{t('settings.runtimeTitle')}</h2>
                    <span className="rounded-full bg-[#1f3a60]/10 px-3 py-1 text-xs font-bold text-[#1f3a60]">{t('settings.runtimeBadge')}</span>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                    <div>
                        <label className="mb-2 block text-sm font-semibold text-slate-700">{t('settings.chatModel')}</label>
                        <select
                            value={runtimeSettings.chat_model}
                            onChange={(event) => setRuntimeSettings((prev) => ({
                                ...prev,
                                chat_model: event.target.value,
                            }))}
                            disabled={!runtimeSettings.available_chat_models?.length}
                            className="h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            {runtimeSettings.available_chat_models?.length ? (
                                runtimeSettings.available_chat_models.map((model) => (
                                    <option key={model} value={model}>{model}</option>
                                ))
                            ) : (
                                <option value="">{t('settings.noChatModels')}</option>
                            )}
                        </select>
                    </div>

                    <div>
                        <label className="mb-2 block text-sm font-semibold text-slate-700">{t('settings.embeddingModel')}</label>
                        <select
                            value={runtimeSettings.embedding_model}
                            onChange={(event) => setRuntimeSettings((prev) => ({
                                ...prev,
                                embedding_model: event.target.value,
                            }))}
                            disabled={!runtimeSettings.available_embedding_models?.length}
                            className="h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            {runtimeSettings.available_embedding_models?.length ? (
                                runtimeSettings.available_embedding_models.map((model) => (
                                    <option key={model} value={model}>{model}</option>
                                ))
                            ) : (
                                <option value="">{t('settings.noEmbeddingModels')}</option>
                            )}
                        </select>
                        <p className="mt-2 text-xs text-slate-500">
                            {t('settings.embeddingHint')}
                        </p>
                        <p className="mt-1 text-xs text-slate-500">
                            {t('settings.embeddingReindexHint')}
                        </p>
                        {!runtimeSettings.ollama_available && (
                            <p className="mt-2 text-xs font-semibold text-amber-600">
                                {t('settings.ollamaFailed', { error: runtimeSettings.ollama_error || t('settings.ollamaFallback') })}
                            </p>
                        )}
                    </div>
                </div>

                <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <label className="flex items-start gap-3">
                        <input
                            type="checkbox"
                            checked={Boolean(runtimeSettings.enable_condense_query)}
                            onChange={(event) => setRuntimeSettings((prev) => ({
                                ...prev,
                                enable_condense_query: event.target.checked,
                            }))}
                            className="mt-1 h-4 w-4 rounded border-slate-300 text-[#1f3a60] focus:ring-[#1f3a60]/25"
                        />
                        <span>
                            <span className="block text-sm font-semibold text-slate-800">{t('settings.condenseQuery')}</span>
                            <span className="mt-1 block text-xs text-slate-500">
                                {t('settings.condenseHint')}
                            </span>
                        </span>
                    </label>
                </div>

                <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <label className="mb-2 block text-sm font-semibold text-slate-700">{t('settings.topK')}</label>
                    <input
                        type="number"
                        min={1}
                        max={20}
                        value={runtimeSettings.top_k ?? 10}
                        onChange={(event) => setRuntimeSettings((prev) => ({
                            ...prev,
                            top_k: Number(event.target.value),
                        }))}
                        className="h-10 w-32 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25"
                    />
                    <p className="mt-2 text-xs text-slate-500">{t('settings.topKHint')}</p>
                </div>

                <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-4">
                    <label className="flex items-start gap-3">
                        <input
                            type="checkbox"
                            checked={Boolean(runtimeSettings.contextual_embedding_enabled)}
                            onChange={(event) => setRuntimeSettings((prev) => ({
                                ...prev,
                                contextual_embedding_enabled: event.target.checked,
                            }))}
                            className="mt-1 h-4 w-4 rounded border-slate-300 text-[#1f3a60] focus:ring-[#1f3a60]/25"
                        />
                        <span>
                            <span className="block text-sm font-semibold text-slate-800">{t('settings.contextualEmbedding')}</span>
                            <span className="mt-1 block text-xs text-slate-500">
                                {t('settings.contextualEmbeddingHint')}
                            </span>
                        </span>
                    </label>

                    <div>
                        <label className="mb-2 block text-sm font-semibold text-slate-700">{t('settings.contextualModel')}</label>
                        <select
                            value={runtimeSettings.contextual_embedding_model}
                            onChange={(event) => setRuntimeSettings((prev) => ({
                                ...prev,
                                contextual_embedding_model: event.target.value,
                            }))}
                            disabled={!runtimeSettings.available_chat_models?.length}
                            className="h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            {runtimeSettings.available_chat_models?.length ? (
                                runtimeSettings.available_chat_models.map((model) => (
                                    <option key={model} value={model}>{model}</option>
                                ))
                            ) : (
                                <option value="">{t('settings.noContextualModels')}</option>
                            )}
                        </select>
                        <p className="mt-2 text-xs text-slate-500">
                            {t('settings.contextualModelHint')}
                        </p>
                    </div>

                    <div>
                        <label className="mb-2 block text-sm font-semibold text-slate-700">
                            Контекстное окно (contextual embedding num_ctx)
                        </label>
                        <input
                            type="number"
                            min={2048}
                            max={32000}
                            step={1024}
                            value={runtimeSettings.contextual_embedding_num_ctx ?? 8192}
                            onChange={(event) => setRuntimeSettings((prev) => ({
                                ...prev,
                                contextual_embedding_num_ctx: Number(event.target.value),
                            }))}
                            className="h-10 w-40 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25"
                        />
                        <p className="mt-1 text-xs text-slate-500">Токенов в контексте при генерации чанков. Рекомендуется 8192.</p>
                    </div>
                </div>

                <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-4">
                    <label className="mb-2 block text-sm font-semibold text-slate-700">
                        Контекстное окно чат-модели (chat num_ctx)
                    </label>
                    <input
                        type="number"
                        min={2048}
                        max={32000}
                        step={1024}
                        value={runtimeSettings.chat_model_num_ctx ?? 20000}
                        onChange={(event) => setRuntimeSettings((prev) => ({
                            ...prev,
                            chat_model_num_ctx: Number(event.target.value),
                        }))}
                        className="h-10 w-40 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25"
                    />
                    <p className="mt-1 text-xs text-slate-500">Токенов в контексте чат-модели. Рекомендуется 12000–20000 (макс. 32000 из-за ограничений памяти).</p>
                </div>

                <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <label className="flex items-start gap-3">
                        <input
                            type="checkbox"
                            checked={Boolean(runtimeSettings.reranker_enabled)}
                            onChange={(event) => setRuntimeSettings((prev) => ({
                                ...prev,
                                reranker_enabled: event.target.checked,
                            }))}
                            className="mt-1 h-4 w-4 rounded border-slate-300 text-[#1f3a60] focus:ring-[#1f3a60]/25"
                        />
                        <span>
                            <span className="block text-sm font-semibold text-slate-800">
                                Reranker — Qwen3-Reranker-4B
                            </span>
                            <span className="mt-1 block text-xs text-slate-500">
                                Переранжирует найденные чанки с помощью cross-encoder модели (HuggingFace). Улучшает точность при похожих документах. Добавляет ~1–3 сек к ответу.
                            </span>
                        </span>
                    </label>
                </div>

                <div className="mt-5 flex flex-wrap items-center gap-3">
                    <Button type="button" onClick={handleSave} isLoading={isSaving} disabled={!canSave}>
                        {t('settings.save')}
                    </Button>
                    {message && <span className="text-sm font-semibold text-emerald-600">{message}</span>}
                    {error && <span className="text-sm font-semibold text-red-600">{error}</span>}
                </div>
            </section>

            <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="border-b border-slate-200 px-5 py-4">
                    <h3 className="text-lg font-bold text-[#1f3a60]">{t('settings.rolesTitle')}</h3>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full min-w-[740px] text-left">
                        <thead className="bg-slate-50 text-xs uppercase tracking-[0.08em] text-slate-500">
                            <tr>
                                <th className="px-5 py-3 font-semibold">{t('settings.table.user')}</th>
                                <th className="px-5 py-3 font-semibold">{t('settings.table.role')}</th>
                                <th className="px-5 py-3 font-semibold">{t('settings.table.createdAt')}</th>
                            </tr>
                        </thead>

                        <tbody>
                            {users.map((user) => (
                                <tr key={user.id} className="border-t border-slate-100 text-sm hover:bg-slate-50">
                                    <td className="px-5 py-3 font-semibold text-slate-800">{user.username}</td>
                                    <td className="px-5 py-3">
                                        <select
                                            value={user.role}
                                            onChange={(event) => handleRoleChange(user.id, event.target.value)}
                                            className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25"
                                        >
                                            {ROLE_OPTIONS.map((role) => (
                                                <option key={role} value={role}>{role}</option>
                                            ))}
                                        </select>
                                    </td>
                                    <td className="px-5 py-3 text-slate-500">
                                        {formatLocaleDate(user.created_at, locale, {
                                            year: 'numeric',
                                            month: 'short',
                                            day: 'numeric',
                                            hour: '2-digit',
                                            minute: '2-digit',
                                        })}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    );
};

export default SettingsPage;
